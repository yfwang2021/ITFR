# coding=utf-8
from copyreg import pickle
import sys
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import datatable as dt

np.random.seed(2022)

RAW_DATA = 'raw data path'
DATA_FILE = os.path.join(RAW_DATA, 'QB-article.csv')
filter_genres = [104, 113, 124, 127]
genres_dict = {104:0, 113:1, 124:2, 127:3}
filter_genders = [1, 2]
filter_ages = [5,6,7]
ages_dict = {5:0, 6:1, 7:2}


def format_user_feature(data):
    print('format_user_feature')
    user_df = data[["user_id", "age"]].drop_duplicates()

    user_df = user_df[user_df.age.isin(filter_ages)]
    user_df["age"] = user_df["age"].apply(lambda x: ages_dict[x])
    user_df.index = user_df["user_id"]
    user_df = user_df.sort_index()
    return user_df


def format_item_feature(data):
    print('format_item_feature')
    item_df = data[["item_id","category_first"]].drop_duplicates()
    
    item_df = item_df[item_df.category_first.isin(filter_genres)]
    item_df['category_first'] = item_df['category_first'].apply(lambda x: genres_dict[x])
    item_df.index = item_df["item_id"]
    item_df = item_df.sort_index()
    return item_df


def format_all_inter(data, user_df, item_df):
    print('format_all_inter')
    
    filter_users = list(user_df["user_id"])
    filter_items = list(item_df["item_id"])
    
    inter_df = data[["user_id", "item_id"]]
    inter_df = inter_df.drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)
    
    inter_df = inter_df[inter_df["user_id"].isin(filter_users) & inter_df["item_id"].isin(filter_items)]
    return inter_df

def random_split_data(all_data_file, val_size=0.1, test_size=0.2):
    all_data = pd.read_csv(all_data_file, sep=' ')
    user_list = list(all_data["user_id"].unique())
    if type(val_size) is float:
        val_size = int(len(all_data) * val_size)
    if type(test_size) is float:
        test_size = int(len(all_data) * test_size)
    validation_set = all_data.sample(n=val_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=test_size).sort_index()
    train_set = all_data.drop(test_set.index)
    
    
    training_dict = {}
    for row in train_set.to_dict(orient="records"):
        if row["user_id"] not in training_dict:
            training_dict[row["user_id"]] = []
        training_dict[row["user_id"]].append(row["item_id"])
        
    validation_dict = {}
    for row in validation_set.to_dict(orient="records"):
        if row["user_id"] not in validation_dict:
            validation_dict[row["user_id"]] = []
        validation_dict[row["user_id"]].append(row["item_id"])
        
    test_dict = {}
    for row in test_set.to_dict(orient="records"):
        if row["user_id"] not in test_dict:
            test_dict[row["user_id"]] = []
        test_dict[row["user_id"]].append(row["item_id"])
    
    for u in user_list:
        if u not in training_dict:
            if u in validation_dict and len(validation_dict[u]) > 1:
                training_dict[u] = []
                training_dict[u].append(validation_dict[u].pop())
            elif u in test_dict and len(test_dict[u]) > 1:
                training_dict[u] = []
                training_dict[u].append(test_dict[u].pop())
        if u not in validation_dict:
            if u in training_dict and len(training_dict[u]) > 1:
                validation_dict[u] = []
                validation_dict[u].append(training_dict[u].pop())
            elif u in test_dict and len(test_dict[u]) > 1:
                validation_dict[u] = []
                validation_dict[u].append(test_dict[u].pop())
        if u not in test_dict:
            if u in training_dict and len(training_dict[u]) > 1:
                test_dict[u] = []
                test_dict[u].append(training_dict[u].pop())
            elif u in validation_dict and len(validation_dict[u]) > 1:
                test_dict[u] = []
                test_dict[u].append(validation_dict[u].pop())
                
    training_dict_len = sum([len(v) for k,v in training_dict.items()])
    validation_dict_len = sum([len(v) for k,v in validation_dict.items()])
    test_dict_len = sum([len(v) for k,v in test_dict.items()])
    
    print('train=%d validation=%d test=%d' % (training_dict_len, validation_dict_len, test_dict_len))
    
    for i, data in enumerate([training_dict, validation_dict, test_dict]):
        user_inter_dict = data
        if i == 0:
            data_path = "../data/tenrec_qba/train.txt"
        elif i == 1:
            data_path = "../data/tenrec_qba/valid.txt"
        elif i == 2:
            data_path = "../data/tenrec_qba/test.txt"
            
        u_ids = sorted(list(user_inter_dict.keys()))
        with open(data_path, "w") as f:
            for u_id in u_ids:
                s = str(u_id)
                for item in user_inter_dict[u_id]:
                    s += " " + str(item)
                s += "\n"
                f.write(s)

    return train_set, validation_set, test_set

def renumber_ids(df, old_column, new_column):
    old_ids = sorted(df[old_column].dropna().astype(int).unique())
    id_dict = dict(zip(old_ids, range(len(old_ids))))
    id_df = pd.DataFrame({new_column: old_ids, old_column: old_ids})
    id_df[new_column] = id_df[new_column].apply(lambda x: id_dict[x])
    id_df.index = id_df[new_column]
    id_df = id_df.sort_index()
    df[old_column] = df[old_column].apply(lambda x: id_dict[x] if x in id_dict else 0)
    df = df.rename(columns={old_column: new_column})
    return df, id_df, id_dict


def main():
    data_dir = os.path.join("../data/", 'tenrec_qba/')
    all_inter_file = os.path.join(data_dir, 'tenrec_qba.all.csv')
    
    data = dt.fread("/home/wangyifan/IntersectionalFairness/data/tenrec/Tenrec/QB-article.csv").to_pandas()
    data = data[data.age.isin(filter_ages)]

    user_df = format_user_feature(data)
    item_df = format_item_feature(data)

    inter_df = format_all_inter(data, user_df, item_df)
    
    inter_df['user_freq'] = inter_df.groupby('user_id')['user_id'].transform('count')
    inter_df['item_freq'] = inter_df.groupby('item_id')['item_id'].transform('count')
    
    least_iter_num = 4
    
    while np.min(inter_df['user_freq']) <= least_iter_num:
        inter_df.drop(inter_df.index[inter_df['user_freq'] <= least_iter_num], inplace=True)
        inter_df.reset_index(drop=True, inplace=True)
        inter_df['item_freq'] = inter_df.groupby('item_id')['item_id'].transform('count')
        inter_df.drop(inter_df.index[inter_df['item_freq'] <= least_iter_num], inplace=True)
        inter_df.reset_index(drop=True, inplace=True)
        inter_df['user_freq'] = inter_df.groupby('user_id')['user_id'].transform('count')
        inter_df.reset_index(drop=True, inplace=True)
        
    inter_df, uid_df, uid_dict = renumber_ids(inter_df, old_column='user_id', new_column="user_id")
    inter_df, iid_df, iid_dict = renumber_ids(inter_df, old_column='item_id', new_column="item_id")
    user_df = user_df[user_df["user_id"].isin(uid_dict)]
    item_df = item_df[item_df["item_id"].isin(iid_dict)]
    user_df["user_id"] = user_df["user_id"].apply(lambda x: uid_dict[x])
    item_df["item_id"] = item_df["item_id"].apply(lambda x: iid_dict[x])
    
    item_df.to_csv('../data/tenrec_qba/tenrec_qba.items.csv', index=False, sep=' ')

    item_group = {}
    for row in item_df.to_dict(orient="records"):
        item_group[row["item_id"]] = row["category_first"]
    pickle.dump(item_group, open("../data/tenrec_qba/item_group.pkl","wb"))
    
    user_df.to_csv('../data/tenrec_qba/tenrec_qba.users.csv', index=False, sep=' ')

    user_group = {}
    for row in user_df.to_dict(orient="records"):
        user_group[row["user_id"]] = row["age"]
        
    pickle.dump(user_group, open("../data/tenrec_qba/user_group.pkl","wb"))
    
    inter_df.to_csv(all_inter_file, sep=' ', index=False)
    train_set, _ ,_ = random_split_data(all_inter_file, val_size=0.1, test_size=0.2)
    return


if __name__ == '__main__':
    main()
