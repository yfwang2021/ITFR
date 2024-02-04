import numpy as np
import random as rd
rd.seed(2022)
np.random.seed(2022)
from time import time
import pickle
import os
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

class Data(Dataset):
    def __init__(self, path, batch_size, dname):
        super(Data).__init__()
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        
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
            
        self.group2items = {}
        for k, v in self.item_group.items():
            if v not in self.group2items:
                self.group2items[v] = []
            self.group2items[v].append(k)

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_valid, self.n_test = 0, 0, 0

        self.exist_users = []
        self.all_inter = []
        self.test_inter = []
        self.group_dict = {}
        group_idx = 0
        for u in self.u_filter:
            for i in self.i_filter:
                self.group_dict["{}_{}".format(u,i)] = group_idx
                group_idx += 1
        self.group_count = [0 for _ in range(group_idx)]
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    
        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')

                    items = [int(i) for i in l.split(' ')[1:]]
                    
                    self.n_items = max(self.n_items, max(items))
                    self.n_valid += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')

                    items = [int(i) for i in l.split(' ')[1:]]

                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.train_items, self.valid_set, self.test_set = {}, {}, {}
        
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        with open(train_file) as f_train:
            with open(valid_file) as f_valid:
                with open(test_file) as f_test:
                    for l in f_train.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, train_items = items[0], items[1:]

                        for i in train_items:
                            self.R[uid, i] = 1.
                            self.all_inter.append([uid, i, self.group_dict["{}_{}".format(self.user_group[uid], self.item_group[i])]])
                            self.group_count[self.group_dict["{}_{}".format(self.user_group[uid], self.item_group[i])]] += 1
                            
                        self.train_items[uid] = train_items
                        
                    for l in f_valid.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')

                        items = [int(i) for i in l.split(' ')]

                        uid, valid_items = items[0], items[1:]
                        self.valid_set[uid] = valid_items
                        
                    for l in f_test.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')

                        items = [int(i) for i in l.split(' ')]
                        
                        uid, test_items = items[0], items[1:]
                        
                        for i in test_items:
                            self.test_inter.append([uid, i, self.group_dict["{}_{}".format(self.user_group[uid], self.item_group[i])]])
                        self.test_set[uid] = test_items
                
        self.print_statistics()
        self.all_inter = np.array(self.all_inter)
        self.neg_items = np.zeros((len(self.all_inter),))
        self.iid_buffer = np.random.randint(low=0, high=self.n_items, size=len(self.all_inter))
        self.buffer_idx = 0 
        self.test_inter = np.array(self.test_inter)
        np.random.shuffle(self.all_inter)
            
        self.idx = 0
        self.test_idx = 0
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test + self.n_valid))
        print('n_train=%d, n_valid=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_valid, self.n_test, (self.n_train + self.n_test + self.n_valid)/(self.n_users * self.n_items)))
        
    def __len__(self):
        return len(self.all_inter)
    
    def __getitem__(self, index):
        user_id, pos_item, group = self.all_inter[index, 0], self.all_inter[index, 1], self.all_inter[index, 2]
        neg_item = self.neg_items[index]
        return { "user_id" : user_id,
                "pos_item": pos_item,
                "neg_item": neg_item,
                "group": group,
                "index": index}
        
    def prepare_neg_sampling(self):
        uids_len = len(self.all_inter)

        exclude_iids = {k: set(self.train_items[k]) for k in self.train_items}

        for i in range(uids_len):
            uid = self.all_inter[i,0]
            exclude = exclude_iids[uid] if uid in exclude_iids else set([])

            neg_item = -1
            while neg_item == -1:
                if len(self.iid_buffer) <= self.buffer_idx:
                    self.iid_buffer = np.random.randint(low=0, high=self.n_items, size=uids_len)
                    self.buffer_idx = 0
                iid = self.iid_buffer[self.buffer_idx]
                self.buffer_idx += 1
                if iid not in exclude:
                    neg_item = iid
            self.neg_items[i] = neg_item