# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:47:40 2022

@author: YWC
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        #print(nhid)
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def _mask(self):
        return self.mask

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
class GCN_one(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        #print(nhid)
        super(GCN_one, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def _mask(self):
        return self.mask

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GAT(nn.Module): #在聚合neighbor的情况下只需一层GAT
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class FeatureFusionGate(nn.Module):
    def __init__(self, embedding_dim):
        super(FeatureFusionGate, self).__init__()
        self.linear1 = nn.Linear(embedding_dim*2, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, bundle_like, item_like):
        concat_feature = torch.cat((bundle_like, item_like), dim=1)
        fusion = F.elu(self.linear1(concat_feature))
        gate = F.sigmoid(self.linear2(concat_feature))
        out = torch.mul(gate, fusion) + torch.mul((1-gate), bundle_like)
        return out



class HGLR(nn.Module):
    def __init__(self, dropout_gnn, dropout_pre, num_user, dim_latent, dim_feat, dim_ori, alpha, hidden_size, active_fun, pred_method):
        super(HGLR, self).__init__()

        self.dropout1 = dropout_gnn
        self.dropout2 = dropout_pre
        self.alpha = alpha
        self.concat = Parameter(torch.ones(2, 1))
        self.softmax = nn.Softmax(dim=0)
        self.preference = nn.init.xavier_normal_(torch.rand((num_user, dim_feat), requires_grad=True)).cuda()
        self.MLP = nn.Linear(dim_ori, dim_feat)

        self.MGCN1 = GCN(dim_feat, dim_latent, dim_latent, self.dropout1)
        self.MGCN2 = GCN_one(dim_feat, dim_latent, self.dropout1)
        self.MGCN3 = GCN_one(dim_feat, dim_latent, self.dropout1)
        self.MGCN4 = GCN_one(dim_feat, dim_latent, self.dropout1)
        self.MGCN5 = GCN_one(dim_feat, dim_latent, self.dropout1)
        self.attention = Attention(dim_latent, hidden_size)
        self.featureFusionGate = FeatureFusionGate(dim_latent)
        self.tanh = nn.Tanh()
        self.result = None

        
        #复杂的预测模块所需
        self.active_fun_str = active_fun
        if self.active_fun_str == 'relu':
            self.active_fun = nn.ReLU()
        else:
            self.active_fun = nn.LeakyReLU()
            
        self.pred_method = pred_method
        if self.pred_method == 'mlp':
            self.predict_l_1 = nn.Linear(in_features=dim_latent * 2, out_features=dim_latent)
            self.predict_l_2 = nn.Linear(in_features=dim_latent, out_features=dim_latent)
            self.predict_l_3 = nn.Linear(in_features=dim_latent, out_features=1)
        elif self.pred_method == 'joint':
            self.predict_l_1 = nn.Linear(in_features=dim_latent * 2, out_features=dim_latent)
            self.predict_l_2 = nn.Linear(in_features=dim_latent, out_features=dim_latent)
            self.predict_ul = nn.Linear(in_features=dim_latent, out_features=dim_latent)
            self.predict_il = nn.Linear(in_features=dim_latent, out_features=dim_latent)
            self.fusion_l = nn.Linear(in_features=dim_latent * 2, out_features=1)
        
        ####### 分界线


    def forward(self, features, features1, fadj, sadj1, sadj2, sadj3, special):
        ori_features = torch.matmul(features,self.softmax(self.concat))
        item_features = ori_features.reshape((195283,-1))
        item_features = self.MLP(item_features)

        bundle_it = self.MGCN5(features1, special)
        item_final = self.featureFusionGate(item_features, bundle_it)
        x = torch.cat((self.preference, item_final), dim=0)
        
        emb1 = self.MGCN1(x, fadj)# basic GCN out1 -- fadj ui biparite graph
        emb2 = self.MGCN2(x, sadj1) # Special_GAT out2 -- sadj1 neighbor graph 1
        emb3 = self.MGCN3(x, sadj2)  # Special_GAT out3 -- sadj2 neighbor graph 2
        emb4 = self.MGCN4(x, sadj3)  # Special_GAT out3 -- sadj3 neighbor graph 3
        
        emb = torch.stack([emb1, emb2, emb3, emb4], dim=1)
        rep, att = self.attention(emb)
        self.result = rep
        return self.result
    
    def lookup_emb(self, emb_index):
        if self.result is not None:
            return self.result[emb_index]
        else:
            return False

    def predict(self, u_emb, i_emb):
        ###### 预测模块
        if self.pred_method == 'mlp':
            drop_layer = nn.Dropout(self.dropout2)
            if self.active_fun_str != 'none':
                layer_1 = self.active_fun(drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
                layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
                predict_value = self.active_fun(self.predict_l_3(layer_2))
            else:
                layer_1 = drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
                layer_2 = drop_layer(self.predict_l_2(layer_1))
                predict_value = self.predict_l_3(layer_2)
        elif self.pred_method == 'joint':
            drop_layer = nn.Dropout(self.dropout2)
            if self.active_fun_str != 'none':
                layer_1 = self.active_fun(drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
                layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
                mat_vec = layer_2
                u_layer = self.active_fun(drop_layer(self.predict_ul(u_emb)))
                i_layer = self.active_fun(drop_layer(self.predict_il(i_emb)))
                rep_vec = torch.mul(u_layer, i_layer)
                predict_value = self.active_fun(self.fusion_l(
                    torch.cat((mat_vec, rep_vec), dim=1)))
            else:
                layer_1 = drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
                layer_2 = drop_layer(self.predict_l_2(layer_1))
                mat_vec = layer_2
                u_layer = drop_layer(self.predict_ul(u_emb))
                i_layer = drop_layer(self.predict_il(i_emb))
                rep_vec = torch.mul(u_layer, i_layer)
                predict_value = self.fusion_l(
                    torch.cat((mat_vec, rep_vec), dim=1))
        else:
            # default dot pruduct
            embd = torch.mm(u_emb, i_emb.t()).sum(0)
            predict_value = embd.flatten()
        return predict_value


