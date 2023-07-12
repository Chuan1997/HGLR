# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:51:54 2022

@author: YWC
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='.../HGLR/data/train1', help='Train dataset path')
    parser.add_argument('--test_path', default='.../HGLR/data/neg1', help='Test dataset path')
    parser.add_argument('--save_file', default='.../HGLR/data/', help='Save path')
    parser.add_argument('--neighbor', default='.../HGLR/data/', help='Neighborhood path')
    parser.add_argument('--exist_file1', default='.../HGLR/data/1_adj_csr_200.npz', help='Path1')
    parser.add_argument('--exist_file2', default='.../HGLR/data/2_adj_csr_200.npz', help='Path2')
    parser.add_argument('--exist_file3', default='.../HGLR/data/3_adj_csr_200.npz', help='Path3')
    parser.add_argument('--text_file1', default='.../HGLR/title-pretrain.npy', help='title')
    parser.add_argument('--text_file2', default='.../HGLR/des-pretrain.npy', help='description')
    parser.add_argument('--pretrain_file1', default='.../HGLR/h_item.npy', help='pretraining')
    parser.add_argument('--bundle_file', default='.../HGLR/data/list_song_map', help='bundle-item')


    parser.add_argument('--l_r', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2', type=float, default=1e-4,  # 1e-4
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--dropout_gnn', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--dropout_pre', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=1800, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number for sampling.')

    parser.add_argument('--dim_feat', type=int, default=64, help='Middle embedding dimension for leanring-128')
    parser.add_argument('--dim_latent', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--hidden', type=int, default=16, help='Dimension for attention.')
    parser.add_argument('--step', type=int, default=2000, help='test batch size.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode for textual information.')
    parser.add_argument('--status', default='exist', help='Whether to load nadj.')
    parser.add_argument('--topKs', type=str, default='[5, 10]')
    parser.add_argument('--pred_method', type=str, default='joint')
    parser.add_argument('--active_fun', type=str, default='none',  # none
                        help='leaky_relu, relu; none means no use.')

    parser.add_argument('--sample', type=int, default=5, help='negative sample size for training')
    parser.add_argument('--range', type=int, default=200, help='neighborhood size')
    parser.add_argument('--listrange', type=int, default=20, help='list threshold')
    parser.add_argument('--nb_heads', type=int, default=8, help='number of head attentions')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
    return parser.parse_args()











