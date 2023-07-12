# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:02:33 2022

@author: YWC
"""

import numpy as np
import scipy.sparse as sp
import time
import random
import torch
import torch.utils.data as data

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def loadDict(preFile):
    fp = open(preFile)
    lines = fp.readlines()
    testDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = int(lineStr[0])
        itemId = int(lineStr[1])
        if userId not in testDict:
            testDict[userId] = []
            testDict[userId].append(itemId)
        else:            
            testDict[userId].append(itemId)
    return testDict

#矩阵写法
def generate_biadj(datapath, n_user, n_item, pkl_path):
    struct_edges = np.genfromtxt(datapath, dtype=np.int32)
    datalist = struct_edges.tolist()
    user_item_dict = loadDict(datapath)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n_user, n_item),dtype=np.float32)
    sadj1 = sp.dok_matrix(sadj)
    x1 = sp.eye(sadj.shape[0]).tolil().rows
    x2 = [x[0] for x in x1]
    x3 = sp.eye(sadj.shape[0]).data[0]
    diag = sp.coo_matrix((x3,(x2,x2)),shape=(n_user, n_item),dtype=np.float32)
    nsadj = normalize(sadj + diag).tocoo()

    start = time.time()
    print('generating adj csr... ')

    rows = np.concatenate((nsadj.row, nsadj.transpose().row + n_user))
    cols = np.concatenate((nsadj.col + n_user, nsadj.transpose().col))
    data = np.ones((nsadj.nnz * 2,))
    adj_csr = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)

    print('saving adj_csr to ' + pkl_path + '/adj_csr.npz')
    sp.save_npz(pkl_path + '/adj_csr.npz', adj_csr)
    print("time elapsed {:.4f}s".format(time.time() - start))

    return datalist, sadj1, adj_csr, user_item_dict


def bundlecom(file,bundle,item,leng):
    rawdata = np.load(file,allow_pickle=True)
    train_mat = sp.dok_matrix((bundle, item), dtype=np.float32)
    for i in range(len(rawdata)):
        for j in rawdata[i][:leng]:
            train_mat[i, j] = 1.0
    return train_mat



class BPRSampling(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRSampling, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if \
				self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, item_i, item_j


def getEmb1(b_list, the_model, item_shift, sample_num=7):
    u_array, i_array = np.array(b_list).transpose()
    i_array += item_shift
    u_array1 = np.repeat(u_array,sample_num)
    i_array1 = np.repeat(i_array,sample_num)
    u_long, i_long = torch.LongTensor(u_array1).cuda(), torch.LongTensor(i_array1).cuda()

    u_emb = the_model.lookup_emb(u_long)
    i_emb = the_model.lookup_emb(i_long)

    return u_emb, i_emb


def getEmb2(b_list, the_model, item_shift):
    u_array, i_array, g_array = np.array(b_list).transpose()
    i_array += item_shift
    u_long, i_long, g_truth = torch.LongTensor(u_array).cuda(), torch.LongTensor(i_array).cuda(), \
                              torch.LongTensor(g_array).cuda()

    u_emb = the_model.lookup_emb(u_long)
    i_emb = the_model.lookup_emb(i_long)

    return u_emb, i_emb, g_truth


# Random matching user negative sampling
def sample_neg(batch_pos_list, p_ui_dic, item_num, sample_num=7):
    batch_neg_list = []
    for u_id, _ in batch_pos_list:
        neg_count = 0
        while True:
            if neg_count == sample_num:
                break
            n_iid = random.randint(0, item_num - 1)
            if n_iid not in p_ui_dic[u_id] and (u_id, n_iid, 0) not in batch_neg_list:
                batch_neg_list.append((u_id, n_iid, 0))
                neg_count += 1
    return batch_neg_list


################## 读取邻居节点信息 ######################
def neadj_generate(path, name, n_user, n_item, nrange, pkl_path, key):
    user_neighbor = {}
    item_neighbor = {}
    user_length = 0
    ab = []
    ac = []
    with open(path + name[0] + '_neigbhor200.txt') as infile:
        line = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            u = int(arr[0])
            v = list(map(float, arr[1:]))
            if u not in user_neighbor:
                user_neighbor[u] = []
            user_neighbor[u].append(v)
            user_length = max(user_length, len(user_neighbor[u]))
            line = infile.readline()

    item_length = 0
    with open(path + name[1] + '_neigbhor200.txt') as infile:
        line = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            v = int(arr[0])
            u = list(map(float, arr[1:]))
            if v not in item_neighbor:
                item_neighbor[v] = []
            item_neighbor[v].append(u)
            item_length = max(item_length, len(item_neighbor[v]))
            line = infile.readline()

    start = time.time()
    for node in range(len(user_neighbor)):
        test = np.zeros((1,n_item))
        test1 = sp.lil_matrix(test)
        find = user_neighbor.get(node, [])[0][:nrange]
        if len(find) > 0:
            find1 = np.array(list(map(int,find)))
        else:
            mid = random.sample(list(range(n_item)),nrange)
            find1 = np.array(mid)
        test1[:,find1] = 1.0
        ab.append(test1)
    df = sp.vstack(ab)
    print("time elapsed {:.4f}s".format(time.time() - start))

    start = time.time()
    for node in range(len(item_neighbor)):
        t = np.zeros((1,n_user))
        t1 = sp.lil_matrix(t)
        f = item_neighbor.get(node, [])[0][:nrange]
        f1 = np.array(list(map(int,f)))
        t1[:,f1] = 1.0
        ac.append(t1)
    df1 = sp.vstack(ac)
    print("time elapsed {:.4f}s".format(time.time() - start))

    rows = np.concatenate((df.row, df1.row + n_user))
    cols = np.concatenate((df.col + n_user, df1.col))
    data = np.ones((df.nnz + df1.nnz,))
    adj_csr = sp.coo_matrix((data, (rows, cols)),shape=(n_user+n_item, n_user+n_item)).tocsr().astype(np.float32)
    print('saving adj_csr to ' + pkl_path + '/{}_adj_csr_{}.npz'.format(str(key),str(nrange)))
    sp.save_npz(pkl_path + '/{}_adj_csr_{}.npz'.format(str(key),str(nrange)), adj_csr)
    
    return adj_csr

def feature(path1, path2, mode):
    t1 = np.load(path1,allow_pickle=True)
    t2 = np.load(path2,allow_pickle=True)
    tt1 = t1.item()
    tt2 = t2.item()
    t_emb1 = []
    for k,j in tt1.items():
        t_emb1.append(j)
    t_emb2 = torch.from_numpy(np.array(t_emb1).squeeze(1))
    
    d_emb1 = []
    for k,j in tt2.items():
        d_emb1.append(j)
    d_emb2 = torch.from_numpy(np.array(d_emb1).squeeze(1))
    
    #mode = 'concat'
    if mode == 'concat':
        fea = torch.cat((t_emb2, d_emb2),dim=1)
        fea_dim = d_emb2.shape[1] * 2
    elif mode == 'add':
        fea = t_emb2 + d_emb2
        fea_dim = d_emb2.shape[1]
    else:
        fea = t_emb2
        fea_dim = d_emb2.shape[1]
    
    return fea, fea_dim


def featureread(path1, path2, path3):
    t1 = np.load(path1, allow_pickle=True)
    t2 = np.load(path2, allow_pickle=True)
    tt1 = t1.item()
    tt2 = t2.item()
    t_emb1 = []
    for k, j in tt1.items():
        t_emb1.append(j)
    t_emb2 = torch.from_numpy(np.array(t_emb1).squeeze(1))

    d_emb1 = []
    for k, j in tt2.items():
        d_emb1.append(j)
    d_emb2 = torch.from_numpy(np.array(d_emb1).squeeze(1))

    fea = torch.cat((t_emb2, d_emb2), dim=0)
    fea_dim = d_emb2.shape[1]

    t3 = np.load(path3, allow_pickle=True)
    i_emb2 = torch.from_numpy(t3)

    return fea.reshape((-1,768,2)).cuda(), fea_dim, i_emb2.cuda()


def loadtestDict(negFile):
    fp = open(negFile)
    lines = fp.readlines()
    testDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = eval(lineStr[0])[0]
        testDict[str(userId)+'_p'] = []
        testDict[str(userId)+'_n'] = []
        testDict[str(userId)+'_p'].append([userId, eval(lineStr[0])[1]])
        for i in lineStr[1:]:
            testDict[str(userId)+'_n'].append([userId, int(i)])
    return testDict



def result_evaluate(user_id: int, top_k_list: list, the_model, test_dict, item_shift):
    one_hit, one_ndcg = [], []
    h_test_items = test_dict[str(user_id) + '_p'].copy()
    test_candidate_items = h_test_items + test_dict[str(user_id) + '_n'].copy()
    # print('True Percent;', len(h_test_items) / len(test_candidate_items))
    random.shuffle(test_candidate_items)
    # Calculate first and then select Top-k
    c_score_list = list()

    u_array, i_array = np.array(test_candidate_items).transpose()
    i_array += item_shift
    u_long, i_long = torch.LongTensor(u_array).cuda(), torch.LongTensor(i_array).cuda()
    te_user_emb = the_model.lookup_emb(u_long)
    te_item_emb = the_model.lookup_emb(i_long)

    test_scores = the_model.predict(te_user_emb, te_item_emb).squeeze().detach().cpu().numpy().tolist()
    t_i = 0
    for _, t_iid in test_candidate_items:
        c_score_list.append([t_iid, test_scores[t_i]])
        t_i += 1
    recommend_list = []
    for ii in range(top_k_list[len(top_k_list) - 1]):
        r_item = -1
        max_score = -np.inf
        for c_score in c_score_list:
            c_item = c_score[0]
            score = c_score[1]
            if score > max_score and c_item not in recommend_list:
                max_score = score
                r_item = c_item
        recommend_list.append(r_item)
    for top_k in top_k_list:
        hit_count = 0
        hit_list = []
        dcg = 0
        idcg = 0
        for k in range(len(recommend_list[:top_k])):
            t_item = recommend_list[k]
            if [user_id, t_item] in h_test_items:
                hit_count += 1
                dcg += 1 / np.log(k + 2)
                hit_list.append(1)
            else:
                hit_list.append(0)
        hit_list.sort(reverse=True)
        kk = 0
        for imp_rating in hit_list:
            idcg += imp_rating / np.log(kk + 2)
            kk += 1
        if hit_count > 0:
            one_hit.append(1)
            one_ndcg.append(dcg / idcg)
        else:
            one_hit.append(0)
            one_ndcg.append(0)
    return one_hit, one_ndcg








