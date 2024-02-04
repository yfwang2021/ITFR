import pickle
import numpy as np

import itertools
import sys
import heapq
import torch
from torch.utils.data import DataLoader

def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)

def recall(rank, ground_truth, item_group, i_group_num, top_k):
    result = np.zeros((top_k, i_group_num), dtype=np.float32)
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            i = ground_truth.index(item)
            result[last_idx:, item_group[i]] += 1.0
    return result

def eval_intersection_result(score_matrix, test_items, top_k=50, batch_item_group=None, i_group_num=0, pre_recall_mat=None, user_batch = None):
    batch_result_recall = []
    for idx in range(len(test_items)):
        scores = score_matrix[idx]  
        test_item = test_items[idx]  
        item_group = batch_item_group[idx]
        user_id = user_batch[idx]

        ranking = argmax_top_k(scores, top_k) 
        
        recall_row = recall(ranking, test_item, item_group, i_group_num, top_k)
        pre_recall_mat_copy = np.copy(pre_recall_mat[user_id])
        pre_recall_mat_copy = np.where(pre_recall_mat_copy > 0, pre_recall_mat_copy, 1e-6)
        recall_row = recall_row/pre_recall_mat_copy
        
        batch_result_recall.append(recall_row)

    return batch_result_recall

class Fairness_Evaluator(object):
    def __init__(self, path, n_users, n_items, dname):
        super(Fairness_Evaluator, self).__init__()
        self.test_file = path + '/test.txt'
        self.train_file = path + '/train.txt'
        self.valid_file = path + "/valid.txt"
        
        user_group = path + '/user_group.pkl'
        item_group = path + '/item_group.pkl'
        
        self.user_group = pickle.load(open(user_group,"rb"))
        self.item_group = pickle.load(open(item_group,"rb"))
        
        if dname == "ml-1m":
            self.u_group_num = 2
            self.i_group_num = 6
            self.u_filter = [0,1]
            self.i_filter = [0,1,2,3,4,5]
        elif dname == "tenrec_qba":
            self.u_group_num = 3
            self.i_group_num = 4
            self.u_filter = [0,1,2]
            self.i_filter = [0,1,2,3]
        elif dname == "lfm2b":
            self.u_group_num = 2
            self.i_group_num = 4
            self.u_filter = [0,1]
            self.i_filter = [0,1,2,3]
        
        self.n_users, self.n_items = n_users, n_items
        
    def load_data(self):
        inter_num = np.zeros((self.u_group_num, self.i_group_num))
        self.valid_set, self.test_set, self.train_items = {}, {}, {}
        with open(self.train_file) as f_train:
            with open(self.valid_file) as f_valid:
                with open(self.test_file) as f_test:
                    for l in f_train.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, train_items = items[0], items[1:]
                            
                        self.train_items[uid] = train_items
                        
                        for item in train_items:
                            inter_num[self.user_group[uid], self.item_group[item]] += 1
                        
                    for l in f_test.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        
                        uid, test_items = items[0], items[1:]
                        self.test_set[uid] = test_items
                        for item in test_items:
                            inter_num[self.user_group[uid], self.item_group[item]] += 1

                    for l in f_valid.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        
                        uid, valid_items = items[0], items[1:]
                        self.valid_set[uid] = valid_items
                        for item in valid_items:
                            inter_num[self.user_group[uid], self.item_group[item]] += 1

        self.pre_recall_mat = np.zeros((self.n_users, self.i_group_num))
        for u in range(self.n_users):
            for i in self.test_set[u]:
                self.pre_recall_mat[u][self.item_group[i]] += 1.0
        
        self.user_group_count = {} # calculate interested users U(i,j)
        for k,v in self.user_group.items():
            if v not in self.user_group_count:
                self.user_group_count[v] = np.zeros(len(self.i_filter))
            for i in range(len(self.i_filter)):
                if self.pre_recall_mat[k][i] > 0:
                    self.user_group_count[v][i] += 1
        
    def evaluate(self, model):
        results = {}
        users_to_test = list(self.test_set.keys())

        top_show = [10,20,50]
        max_top = max(top_show)
        
        u_batch_size = 1024

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        
        intersection_recall_matrix = np.zeros((len(top_show), self.u_group_num, self.i_group_num))

        item_batch = range(self.n_items)
        
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]
            batch_user_group = []
            for u_id in user_batch:
                batch_user_group.append(self.user_group[u_id])
            
            model._pre(user_batch, item_batch)
            rate_batch = model.batch_ratings
            rate_batch = np.array(rate_batch.detach().cpu())
                
            test_items = []
            batch_item_group = []

            for user in user_batch:
                test_items.append(self.test_set[user])
                    
                temp = []
                for item in test_items[-1]:
                    temp.append(self.item_group[item])
                batch_item_group.append(temp)
                 
            for idx, user in enumerate(user_batch):
                train_items_off = self.train_items[user]
                rate_batch[idx][train_items_off] = -np.inf
                valid_items_off = self.valid_set[user]
                rate_batch[idx][valid_items_off] = -np.inf
            
            batch_recall_result = eval_intersection_result(rate_batch, test_items, max_top, batch_item_group, self.i_group_num, self.pre_recall_mat, user_batch) #[user, topk, item_group]
            
            for u_idx, user in enumerate(batch_user_group):
                for i, top in enumerate(top_show):
                    intersection_recall_matrix[i, user] += batch_recall_result[u_idx][top - 1]

        for u_key in self.u_filter:
            intersection_recall_matrix[:,u_key] = intersection_recall_matrix[:,u_key] / self.user_group_count[u_key]
            
        recall_list = []
        for i in range(3):
            l = []
            for u_key in self.u_filter:
                l += [j for j in intersection_recall_matrix[i][u_key] if j > 0]
            recall_list.append(l)
            
        assert(len(recall_list[1]) == self.u_group_num*self.i_group_num)
        results['recall_matrix_var'] = [np.std(l)/np.mean(l) for l in recall_list]
        results['recall_matrix_min_25'] = [np.mean(np.sort(l)[:len(l)//4]) for l in recall_list]
        
        def cal_u_side(recall_list):
            u_side = []
            for l in recall_list:
                mat = np.array(l).reshape(self.u_group_num, self.i_group_num)
                u_cov = 0
                for i in range(self.i_group_num):
                    u_cov += np.std(mat[:,i])/np.mean(mat[:,i])
                u_cov = u_cov / self.i_group_num
                u_side.append(u_cov)
            return u_side

        def cal_i_side(recall_list):
            i_side = []
            for l in recall_list:
                mat = np.array(l).reshape(self.u_group_num, self.i_group_num)
                i_cov = 0
                for i in range(self.u_group_num):
                    i_cov += np.std(mat[i, :])/np.mean(mat[i, :])
                i_cov = i_cov / self.u_group_num
                i_side.append(i_cov)
            return i_side
        
        results['recall_matrix_uside'] = cal_u_side(recall_list=recall_list)
        results['recall_matrix_iside'] = cal_i_side(recall_list=recall_list)
        return results