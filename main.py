# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:11:53 2022

@author: YWC
"""

import argparse
import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from datetime import datetime
import random
from Dataset import normalize, sparse_mx_to_torch_sparse_tensor, generate_biadj, neadj_generate, featureread, getEmb1, getEmb2, sample_neg, loadtestDict, result_evaluate, bundlecom
from model import HGLR

from params import parse_args

###############################248###########################################

args = parse_args()

class RunProcess:
    def __init__(self, args, pathname, user, item, num):
        self.args = args
        self.model_state = None
        # Load data
        print('Data loading ...')
        self.train_data, self.single_adj, self.combined_adj, self.user_item_dict = generate_biadj(self.args.data_path, user, item, self.args.save_file)
        
        self.fadj = normalize(self.combined_adj + sp.eye(self.combined_adj.shape[0]))
        self.fadj_norm = sparse_mx_to_torch_sparse_tensor(self.fadj).cuda()
        
        if self.args.status == 'not exist':
            self.sadj1 = neadj_generate(self.args.neighbor, pathname[0:2], user, item, self.args.range, self.args.save_file, 1)
            self.sadj1_st = sparse_mx_to_torch_sparse_tensor(self.sadj1 + sp.eye(self.sadj1.shape[0])).cuda()
            
            self.sadj2 = neadj_generate(self.args.neighbor, pathname[2:4], user, item, self.args.range, self.args.save_file, 2)
            self.sadj2_st = sparse_mx_to_torch_sparse_tensor(self.sadj2 + sp.eye(self.sadj2.shape[0])).cuda()
           
            self.sadj3 = neadj_generate(self.args.neighbor, pathname[4:6], user, item, self.args.range, self.args.save_file, 3)
            self.sadj3_st = sparse_mx_to_torch_sparse_tensor(self.sadj3 + sp.eye(self.sadj3.shape[0])).cuda()
        else:
            self.sadj1 = sp.load_npz(self.args.exist_file1)
            self.sadj1_st = sparse_mx_to_torch_sparse_tensor(self.sadj1 + sp.eye(self.sadj1.shape[0])).cuda()
            
            self.sadj2 = sp.load_npz(self.args.exist_file2)
            self.sadj2_st = sparse_mx_to_torch_sparse_tensor(self.sadj2 + sp.eye(self.sadj2.shape[0])).cuda()
            
            self.sadj3 = sp.load_npz(self.args.exist_file3)
            self.sadj3_st = sparse_mx_to_torch_sparse_tensor(self.sadj3 + sp.eye(self.sadj3.shape[0])).cuda()

        print('loading')
        self.ibud = bundlecom(self.args.bundle_file, item, num, self.args.listrange)
        self.ibud_st = sparse_mx_to_torch_sparse_tensor(self.ibud).cuda()
        print('partially done')

        self.fea, self.fea_dim, self.itemf = featureread(self.args.text_file1, self.args.text_file2, self.args.pretrain_file1)
        self.test_data = loadtestDict(self.args.test_path)
        print('Data has been loaded.')
        
        self.item_shift = user
        self.sample_item = item
        self.sample_user = user
        self.topKs = eval(self.args.topKs)
        self.regs = eval(self.args.regs)
        self.decay = self.regs[0]

        self.train_p_list = self.train_data
        print('Positive train data:', len(self.train_p_list))

        # Create Model
        self.model = HGLR(self.args.dropout_gnn, self.args.dropout_pre, user, item, self.args.dim_latent, self.args.dim_feat, self.fea_dim,
                             self.args.alpha, self.args.nb_heads, self.args.hidden, self.args.active_fun, self.args.pred_method)
        self.model.cuda()
        # optimizer weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.l_r, weight_decay=self.args.l2)
    ##########################################################################################################################################

    def train(self):
        # Start training
        self.model.train()
        self.model(self.fea, self.itemf, self.fadj_norm, self.sadj1_st, self.sadj2_st, self.sadj3_st, self.ibud_st)
        net_dropout = self.args.dropout_pre

        batch_pos_list = random.sample(self.train_p_list, self.args.batch_size // 2)
        batch_neg_list = sample_neg(batch_pos_list, self.user_item_dict, self.sample_item)
        p_user_emb, p_item_emb = getEmb1(batch_pos_list, self.model, self.item_shift)
        pos_scores = self.model.predict(p_user_emb, p_item_emb)
        n_user_emb, n_item_emb, _ = getEmb2(batch_neg_list, self.model, self.item_shift)
        neg_scores = self.model.predict(n_user_emb, n_item_emb)
        loss = (-nn.LogSigmoid()(pos_scores - neg_scores)).sum()

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, best_hit, best_ndcg):
        list_hits, list_ndcgs = [], []
        for _ in self.topKs:
            list_hits.append([])
            list_ndcgs.append([])
        with torch.no_grad():
            self.model.eval()
            for t_uid in range(self.sample_user):
                try:
                    one_hit, one_ndcg = result_evaluate(
                        t_uid, self.topKs, self.model, self.test_data, self.item_shift)
                    kk = 0
                    for _ in self.topKs:
                        list_hits[kk].append(one_hit[kk])
                        list_ndcgs[kk].append(one_ndcg[kk])
                        kk += 1
                except KeyError:
                    continue

            kk = 0
            str_log = ''
            for top_k in self.topKs:
                if len(list_hits) > 0:
                    t_hit = np.array(list_hits[kk]).mean()
                    t_ndcg = np.array(list_ndcgs[kk]).mean()
                else:
                    t_hit = 0
                    t_ndcg = 0
                best_hit[kk] = best_hit[kk] if best_hit[kk] > t_hit else t_hit
                best_ndcg[kk] = best_ndcg[kk] if best_ndcg[kk] > t_ndcg else t_ndcg
                str_log += 'Top %3d: ' \
                           'HR = %.6f, NDCG = %.6f \n' \
                           'Best HR = %.6f, NDCG = %.6f\n' % \
                           (top_k, t_hit, t_ndcg, best_hit[kk], best_ndcg[kk])
                kk += 1
        return str_log


if __name__ == '__main__':
    if not os.path.exists('.../HGLR/result_log/'):
        os.makedirs('.../HGLR/result_log/')
    result_log = open('.../HGLR/result_log/' + 'HGLR'+ '--'
                      + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w')
    result_log.write(str(args) + '\n')
    result_log.flush()

    print(args)
    name = ['ugl','lgu','ugul','lugu','ulgl','lglu']
    n_user = 14833
    n_item = 195283
    item_num = 3693730
    process = RunProcess(args, name, n_user, n_item, item_num)

    best_hit, best_ndcg = [], []
    for _ in eval(args.topKs):
        best_hit.append(0)
        best_ndcg.append(0)
    for epoch in range(args.num_epoch + 1):
        t_loss = process.train()
        if epoch % 10 == 0:
            train_log = "Epoch: {epoch}, Training Loss: {loss}".format(epoch=epoch, loss=str(t_loss))
            test_log = process.test(best_hit, best_ndcg)
            all_log = train_log + '\n' + test_log + '\n'
            print(all_log)
            result_log.write(all_log)
            result_log.flush()
    result_log.close()


