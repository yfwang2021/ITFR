import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import numpy as np
import random
from utility.helper import *
from utility.batch_test import *

torch.manual_seed(2022 + args.seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(2022 + args.seed)
np.random.seed(2022 + args.seed)
random.seed(2022 + args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

import sys
from utility.intersection_evaluator import Fairness_Evaluator
from copy import deepcopy
from torch.utils.data import DataLoader

class MF(nn.Module):
    def __init__(self, data_config):
        super(MF, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        self.weights = self._init_weights()
        self.group_num = len(data_generator.group_dict)

        self.g_counts = data_generator.group_count
        self.g_counts = torch.tensor(self.g_counts, dtype=torch.float32).cuda()
        wts = torch.ones_like(self.g_counts)
        self.wts = wts/wts.sum()
        self.alpha = torch.autograd.Variable(torch.ones(self.group_num)*(1./self.group_num), requires_grad=False)
        self.rwt = torch.autograd.Variable(self.wts.clone(), requires_grad=False)
        self.step_size = args.eta
        self.sub_loss_list = [0 for _ in range(len(data_generator.group_dict))]

        self.Ks = eval(args.Ks)
        self.epoch_grad_before = [None]*self.group_num
        self.epoch_grad_after = [None]*self.group_num
        self.RTG = torch.zeros([self.group_num, self.group_num]).cuda()
        self.cum_loss_list = [[] for _ in range(len(data_generator.group_dict))]
        self.cum_num_list = [[] for _ in range(len(data_generator.group_dict))]
        self.loss_after = [0 for _ in range(len(data_generator.group_dict))]
        for i in range(self.group_num):
            self.RTG[i,i] = 1
        

    def _pre(self, users, pos_items):
        users = torch.tensor(users, dtype=torch.long).to("cuda")
        pos_items = torch.tensor(pos_items, dtype=torch.long).to("cuda")
        with torch.no_grad():
            u_e = torch.index_select(self.weights["user_embedding"], 0, users)
            pos_i_e = torch.index_select(self.weights["item_embedding"], 0, pos_items)
            u_e = torch.nn.functional.normalize(u_e, dim=-1)
            pos_i_e = torch.nn.functional.normalize(pos_i_e, dim=-1)
            batch_ratings = torch.matmul(u_e, pos_i_e.T)
            self.batch_ratings = batch_ratings

    def _init_weights(self):
        all_weights = nn.ParameterDict()
        all_weights['user_embedding'] = nn.Parameter(torch.zeros([self.n_users, self.emb_dim]))
        all_weights['item_embedding'] = nn.Parameter(torch.zeros([self.n_items, self.emb_dim]))
        nn.init.xavier_uniform_(all_weights['user_embedding'])
        nn.init.xavier_uniform_(all_weights['item_embedding'])

        return all_weights
    
    def epoch_end_process(self):
        for g in range(self.group_num):
            self.loss_after[g] = np.average(self.cum_loss_list[g], weights=self.cum_num_list[g]).astype(np.float)
        self.epoch_grad_before = self.epoch_grad_after
        self.epoch_grad_after = [None]*self.group_num
        self.cum_loss_list = [[] for _ in range(len(data_generator.group_dict))]
        self.cum_num_list = [[] for _ in range(len(data_generator.group_dict))]

if __name__ == '__main__':    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    evaluator = Fairness_Evaluator(path=args.data_path + args.dataset, n_users=data_generator.n_users, n_items=data_generator.n_items, dname=args.dataset)
    evaluator.load_data()

    t0 = time()

    model = MF(data_config=config).to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr = model.lr, weight_decay=eval(args.regs)[0])
    param_groups = optim.param_groups

    cur_best_pre_0 = 0.
    stopping_step = 0
    best_model = None
    best_epoch = -1
    state = {}
    for epoch in range(args.epoch):
        t1 = time()
        loss = 0.

        data_generator.prepare_neg_sampling()
        dataloader = DataLoader(data_generator, batch_size=args.batch_size, shuffle=True, num_workers=3)
        for batch in dataloader:
            batch_group_idx = [0 for _ in range(model.group_num)]
            
            users, pos_items, neg_items, groups = batch["user_id"], batch["pos_item"], batch["neg_item"], batch["group"]
            users = users.long().to("cuda")
            pos_items = pos_items.long().to("cuda")
            neg_items = neg_items.long().to("cuda")
            groups = groups.long().to("cuda")

            flag = 0
            for g in range(model.group_num):
                gidx = torch.nonzero(groups == g, as_tuple=False)
                batch_group_idx[g] = gidx
                if gidx.shape[0] == 0:
                    flag = 1
                    
            sub_loss_list = []
            
            all_grads = [None]*model.group_num
            # calculate sharpness-aware gradient for each intersectional group
            for li in range(model.group_num):
                if batch_group_idx[li].shape[0] == 0:
                    sub_loss_list.append(0)
                    continue
                
                gidx = batch_group_idx[li].squeeze()
                
                group_users = torch.index_select(model.weights["user_embedding"], 0, users[gidx])
                group_pos_items = torch.index_select(model.weights["item_embedding"], 0, pos_items[gidx])
                group_neg_items = torch.index_select(model.weights["item_embedding"], 0, neg_items[gidx])
                
                group_users = torch.nn.functional.normalize(group_users, dim=-1)
                group_pos_items = torch.nn.functional.normalize(group_pos_items, dim=-1)
                group_neg_items = torch.nn.functional.normalize(group_neg_items, dim=-1)
                
                pos_scores = (group_users * group_pos_items).sum(axis=1)
                neg_scores = (group_users * group_neg_items).sum(axis=1)
                
                pos_scores = pos_scores / args.tau
                neg_scores = neg_scores / args.tau

                maxi = -torch.log(nn.Sigmoid()(pos_scores - neg_scores)).mean()
                
                optim.zero_grad()
                maxi.backward()
                
                if gidx.dim() > 0:
                    num_size = len(gidx)
                else:
                    num_size = 1
                par = param_groups[0]["params"]
                if model.epoch_grad_after[li] is None:
                    model.epoch_grad_after[li] = [] 
                    for pi in range(len(par)):
                        model.epoch_grad_after[li].append(par[pi].grad.detach().clone().flatten()*num_size/100)
                else:
                    for pi in range(len(par)):
                        model.epoch_grad_after[li][pi] += par[pi].grad.detach().clone().flatten()*num_size/100
                

                # first-order Taylor approximation for sharpness
                with torch.no_grad():
                    grad_norm = torch.norm(torch.stack([(p.grad).norm(p=2) for group in param_groups for p in group["params"] if p.grad is not None]), p=2)
                    for group in param_groups:
                        scale = args.rho / (grad_norm + 1e-12)

                        for p in group["params"]:
                            if p.grad is None: continue
                            if li == 0:
                                state[p] = p.data.clone()
                            e_w = p.grad * scale.to(p)
                            p.add_(e_w)

                group_users = torch.index_select(model.weights["user_embedding"], 0, users[gidx])
                group_pos_items = torch.index_select(model.weights["item_embedding"], 0, pos_items[gidx])
                group_neg_items = torch.index_select(model.weights["item_embedding"], 0, neg_items[gidx])
                
                group_users = torch.nn.functional.normalize(group_users, dim=-1)
                group_pos_items = torch.nn.functional.normalize(group_pos_items, dim=-1)
                group_neg_items = torch.nn.functional.normalize(group_neg_items, dim=-1)
                
                pos_scores = (group_users * group_pos_items).sum(axis=1)
                neg_scores = (group_users * group_neg_items).sum(axis=1)
                
                pos_scores = pos_scores / args.tau
                neg_scores = neg_scores / args.tau

                maxi = -torch.log(nn.Sigmoid()(pos_scores - neg_scores)).mean()
                sub_loss_list.append(maxi.detach())
                model.cum_loss_list[li].append(maxi.detach().cpu().numpy())
                if gidx.dim() > 0:
                    model.cum_num_list[li].append(len(gidx))
                else:
                    model.cum_num_list[li].append(1)
                    
                params = list(model.parameters())   
                all_grads[li] = torch.autograd.grad(maxi, params)
                assert all_grads[li] is not None

                with torch.no_grad():
                    for group in param_groups:
                        for p in group["params"]:
                            if p.grad is None: continue
                            p.data = state[p]
            
            # calculate collaborative fair weight for each intersectional group
            if flag == 0:    
                if epoch > 0:
                    model.RTG = torch.zeros([model.group_num, model.group_num]).cuda()
                    for li in range(model.group_num):
                        for lj in range(model.group_num):
                            if li == lj:
                                model.RTG[li, lj] = 1
                                continue

                            dp = 0
                            vec1_sqnorm, vec2_sqnorm = 0, 0
                            for pi in range(len(params)):
                                fvec1 = model.epoch_grad_before[lj][pi]
                                fvec2 = all_grads[li][pi].detach().flatten()
                                dp += fvec1 @ fvec2
                                vec1_sqnorm += torch.norm(fvec1)**2
                                vec2_sqnorm += torch.norm(fvec2)**2
                            model.RTG[li, lj] = dp/torch.clamp(torch.sqrt(vec1_sqnorm*vec2_sqnorm), min=1e-12)
                                

                _gl = torch.sqrt(torch.tensor(sub_loss_list).cuda().detach().unsqueeze(-1))
                
                if epoch == 0:
                    RTG = torch.mm(_gl, _gl.t()) * model.RTG
                else:
                    loss_m = torch.mm(_gl, torch.sqrt(torch.tensor(model.loss_after).float().cuda().unsqueeze(-1).t()))
                    for gi in range(model.group_num):
                        loss_m[gi, gi] = sub_loss_list[gi]
                    RTG = loss_m * model.RTG
                    
                wts = torch.tensor(sub_loss_list).cuda()
                wts = wts ** args.gamma
                wts = wts/ wts.sum()
                _exp = model.step_size*(RTG @ wts)
                
                _exp -= _exp.max()
                model.alpha.data = torch.exp(_exp)
                model.rwt *= model.alpha.data
                model.rwt = model.rwt/model.rwt.sum()
                model.rwt = torch.clamp(model.rwt, min=1e-5)
        
            optim.zero_grad()
            params = list(model.parameters()) 
            with torch.no_grad():
                for li in range(model.group_num): 
                    if all_grads[li] is None: continue
                    for pid in range(len(params)):
                        params[pid].grad += model.rwt[li]*all_grads[li][pid]
            optim.step()

            if flag == 0:
                loss += sum(sub_loss_list)

        if np.isnan(loss.detach().cpu()) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % args.verbose != 0:
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs]: train==%.5f' % (epoch, time()-t1, loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret_test = test(model, users_to_test, valid=False)
        
        users_to_valid = list(data_generator.valid_set.keys())
        ret_valid = test(model, users_to_valid, valid=True)
        
        t3 = time()

        if args.verbose > 0:
            perf_str = 'Valid Epoch %d [%.1fs + %.1fs]: train==[%.5f], recall=[%.5f], ' \
                    'precision=[%.5f], ndcg=[%.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, ret_valid['recall'][1], ret_valid['precision'][1], ret_valid['ndcg'][1])
            print(perf_str)
            perf_str = 'Test Epoch %d [%.1fs + %.1fs]: train==[%.5f], recall=[%.5f], ' \
                    'precision=[%.5f], ndcg=[%.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, ret_test['recall'][1], ret_test['precision'][1], ret_test['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret_valid['ndcg'][1], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.patience)
        
        if stopping_step == 0:
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
            
        if should_stop == True:
            break
        
        model.epoch_end_process()
            
    model.load_state_dict(best_model)

    results = evaluator.evaluate(model)
    
    users_to_test = list(data_generator.test_set.keys())
    ret_test = test(model, users_to_test, valid=False)
    
    users_to_valid = list(data_generator.valid_set.keys())
    ret_valid = test(model, users_to_valid, valid=True)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%.5f], precision=[%.5f], ndcg=[%.5f], CV@20=[%.5f], MIN@20=[%.5f], UCV@20=[%.5f], ICV@20=[%.5f]" % \
                (best_epoch, time() - t0, ret_test['recall'][1], ret_test['precision'][1], ret_test['ndcg'][1],
                 results["recall_matrix_var"][1], results["recall_matrix_min_25"][1], results["recall_matrix_uside"][1], results["recall_matrix_iside"][1])
    print(final_perf)

