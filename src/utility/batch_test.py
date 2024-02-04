from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
cores = multiprocessing.cpu_count() // 2

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, dname=args.dataset)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_VALID, N_TEST = data_generator.n_train, data_generator.n_valid, data_generator.n_test

BATCH_SIZE = args.batch_size

def test(model, users_to_test, valid=False):
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {'precision': np.zeros((len(model.Ks),)), 'recall': np.zeros((len(model.Ks),)), 'ndcg': np.zeros((len(model.Ks),))}

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    count = 0
    all_result = []
    item_batch = range(ITEM_NUM)
    
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        
        model._pre(user_batch, item_batch)
        rate_batch = model.batch_ratings
        
        rate_batch = np.array(rate_batch.detach().cpu()).copy()
        test_items = []

        for user in user_batch:
            if valid == False:
                test_items.append(data_generator.test_set[user])
            elif valid == True:
                test_items.append(data_generator.valid_set[user])
                
        for idx, user in enumerate(user_batch):
            train_items_off = data_generator.train_items[user]
            rate_batch[idx][train_items_off] = -np.inf
        
        if valid == False:
            for idx, user in enumerate(user_batch):
                valid_items_off = data_generator.valid_set[user]
                rate_batch[idx][valid_items_off] = -np.inf
        
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top, 1)
        count += len(batch_result)
        all_result.append(batch_result)
    
    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)
    final_result = np.reshape(final_result, newshape=[3, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[3, len(top_show)])
    
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[2]
    return result