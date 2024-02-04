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
import json
import datatable as dt

np.random.seed(2022)

RAW_DATA = 'raw data path'
RATINGS_FILE = os.path.join(RAW_DATA, 'micro-events.tsv')
USERS_FILE = os.path.join(RAW_DATA, 'users.tsv')
ITEMS_FILE = os.path.join(RAW_DATA, 'tags-micro-genres.json')
filter_genres = ["rock", "pop", "jazz", "ambient"]


def format_user_feature(out_file):
    print('format_user_feature', USERS_FILE, out_file)
    user_df = pd.read_csv(USERS_FILE, sep='\t')
    user_df = user_df[["user_id", "age", "gender"]]
    user_df.columns = ["u_id_c", 'u_age_c', 'u_gender_c']

    user_df['u_age_c'] = user_df['u_age_c'].apply(
        lambda x: 0 if x < 18 else (((x + 5) // 10) - 1) if x < 45 else 4 if x < 50 else 5 if x < 56 else 6)

    user_df['u_gender_c'] = user_df['u_gender_c'].apply(lambda x: defaultdict(int, {'m': 0, 'f': 1, 'n' : 2})[x])
    
    user_df = user_df[user_df.u_gender_c.isin([0, 1])]
    
    user_df.index = user_df["u_id_c"]
    user_df = user_df.sort_index()
    return user_df


def format_item_feature(out_file):
    print('format_item_feature', ITEMS_FILE, out_file)

    genres = []
    with open(ITEMS_FILE, 'r') as file:
        for line in file.readlines():
            genres.append(json.loads(line))
            
    ids = []
    first_genres = []
    for song in genres:
        ids.append(song['i'])
        first_genres.append(list(song['tags'].keys())[0])
        
    item_df = pd.DataFrame({"i_id_c": ids, "first_genres": first_genres})

    item_df = item_df[item_df.first_genres.isin(filter_genres)]
    item_df = item_df.sample(frac=0.1, random_state=2022)
    item_df.index = item_df["i_id_c"]
    item_df = item_df.sort_index()
    return item_df


def format_all_inter(out_file, user_df, item_df, label01=False):
    print('format_all_inter', RATINGS_FILE, out_file)
    
    filter_items = list(item_df['i_id_c'])
    filter_users = list(user_df['u_id_c'])
    
    inter_df = dt.fread(RATINGS_FILE, sep='\t').to_pandas()
    inter_df.columns = ["u_id_c", "i_id_c", "album_id", "time"]
    inter_df = inter_df.sort_values(by=["time", "u_id_c"], kind='mergesort')
    inter_df = inter_df.drop_duplicates(["u_id_c", "i_id_c"], keep="last").reset_index(drop=True)
    
    inter_df["year"] = inter_df.apply(lambda x: x["time"].year
                                      ,axis=1)
    inter_df = inter_df[inter_df["year"] > 2014]
    inter_df = inter_df[inter_df["i_id_c"].isin(filter_items)]
    inter_df = inter_df[inter_df["u_id_c"].isin(filter_users)]
    return inter_df

def random_split_data(all_data_file, dataset_name, val_size=0.1, test_size=0.2):
    all_data = pd.read_csv(all_data_file, sep='\t')
    user_list = list(all_data["u_id_c"].unique())
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
        if row["u_id_c"] not in training_dict:
            training_dict[row["u_id_c"]] = []
        training_dict[row["u_id_c"]].append(row["i_id_c"])
        
    validation_dict = {}
    for row in validation_set.to_dict(orient="records"):
        if row["u_id_c"] not in validation_dict:
            validation_dict[row["u_id_c"]] = []
        validation_dict[row["u_id_c"]].append(row["i_id_c"])
        
    test_dict = {}
    for row in test_set.to_dict(orient="records"):
        if row["u_id_c"] not in test_dict:
            test_dict[row["u_id_c"]] = []
        test_dict[row["u_id_c"]].append(row["i_id_c"])
    
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
            data_path = "../data/lfm2b/train.txt"
        elif i == 1:
            data_path = "../data/lfm2b/valid.txt"
        elif i == 2:
            data_path = "../data/lfm2b/test.txt"
            
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
    data_dir = os.path.join("../data/", 'lfm2b/')

    user_file = os.path.join(data_dir, 'users.tsv')
    user_df = format_user_feature(user_file)

    item_file = os.path.join(data_dir, 'items.tsv')
    item_df = format_item_feature(item_file)

    all_inter_file = os.path.join(data_dir, 'inters.tsv')
    inter_df = format_all_inter(all_inter_file, user_df, item_df, label01=True)
    dataset_name = 'lfm2b'
    
    inter_df['user_freq'] = inter_df.groupby('u_id_c')['u_id_c'].transform('count')
    inter_df['item_freq'] = inter_df.groupby('i_id_c')['i_id_c'].transform('count')
    
    least_freq = 9
    while np.min(inter_df['user_freq']) <= least_freq:
        inter_df.drop(inter_df.index[inter_df['user_freq'] <= least_freq], inplace=True)
        inter_df.reset_index(drop=True, inplace=True)
        inter_df['item_freq'] = inter_df.groupby('i_id_c')['i_id_c'].transform('count')
        inter_df.drop(inter_df.index[inter_df['item_freq'] <= least_freq], inplace=True)
        inter_df.reset_index(drop=True, inplace=True)
        inter_df['user_freq'] = inter_df.groupby('u_id_c')['u_id_c'].transform('count')
        inter_df.reset_index(drop=True, inplace=True)
        
    inter_df, uid_df, uid_dict = renumber_ids(inter_df, old_column='u_id_c', new_column="u_id_c")
    inter_df, iid_df, iid_dict = renumber_ids(inter_df, old_column='i_id_c', new_column="i_id_c")
    user_df = user_df[user_df["u_id_c"].isin(uid_dict)]
    item_df = item_df[item_df["i_id_c"].isin(iid_dict)]
    user_df["u_id_c"] = user_df["u_id_c"].apply(lambda x: uid_dict[x])
    item_df["i_id_c"] = item_df["i_id_c"].apply(lambda x: iid_dict[x])
    
    item_df.to_csv('../data/lfm2b/items.tsv', index=False, sep='\t')

    item_group = {}
    for row in item_df.to_dict(orient="records"):
        for i, genre in enumerate(filter_genres):
            if row['first_genres'] == genre:
                item_group[row["i_id_c"]] = i 
    pickle.dump(item_group, open("../data/lfm2b/item_group.pkl","wb"))

    user_df.to_csv('../data/lfm2b/users.tsv', index=False, sep='\t')

    user_group = {}
    for row in user_df.to_dict(orient="records"):
        user_group[row["u_id_c"]] = row["u_gender_c"]
        
    pickle.dump(user_group, open("../data/lfm2b/user_group.pkl","wb"))
    
    inter_df.to_csv(all_inter_file, sep='\t', index=False)
    train_set, _ ,_ = random_split_data(all_inter_file, dataset_name, val_size=0.1, test_size=0.2)
    return


if __name__ == '__main__':
    main()
