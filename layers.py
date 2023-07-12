# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:47:01 2022

@author: YWC
"""

import math
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max
import torch.sparse as sparse

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        """
               :param in_features:     size of the input per node
               :param out_features:    size of the output per node
               :param bias:            whether to add a learnable bias before the activation
               :param device:          device used for computation
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttention(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, out_dim, dropout, alpha, concat=False):
        super(SparseGATLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        N = x.size()[0]   # 图中节点数
        edge = adj._indices()   # 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        if x.is_sparse:   # 判断特征是否为稀疏矩阵
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        #横向拼接
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        values = self.attn.mm(edge_h).squeeze()   # 使用注意力参数对特征进行投射
        edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge，对应原论文中的添加leakyrelu操作
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        e_rowsum = spmm(edge, edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).cuda())  # e_rowsum: N x 1，spmm是稀疏矩阵和非稀疏矩阵的乘法操作
        h_prime = spmm(edge, edge_e, n=N,m=N, matrix=h)   # 把注意力评分与每个节点对应的特征相乘
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())  # h_prime: N x out，div一看就是除，并且每一行的和要加一个9e-15防止除数为0
        # softmax结束
        if self.concat:
            # if this layer is not last layer
            return F.elu(h_prime)
        else:
            # if this layer is last layer
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
