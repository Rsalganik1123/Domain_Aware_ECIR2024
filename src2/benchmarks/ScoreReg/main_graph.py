import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
import pickle 
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
 

# from tensorboardX import SummaryWriter

from src2.benchmarks.ScoreReg.model_graph import NGCF, NGCF_MACR, LightGCN, LightGCN_MACR
#from time import time

#import model
from src2.benchmarks.ScoreReg.config import * 
from  src2.benchmarks.ScoreReg.evaluate import * 
from src2.benchmarks.ScoreReg.data_utils import * 


import scipy.sparse as sp

from src2.benchmarks.ScoreReg.pop_bias_metrics_graph import pred_item_rank, pred_item_score, pred_item_stdscore, pred_item_rankdist, raw_pred_score, pcc_train, pcc_test, pcc_test_check, uPO
import scipy.stats as stats

from scipy.stats import skew


import random as random
random.seed(0)

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cudnn.benchmark = True

 
def train(args, data_path, pretrained_path = None, epoch_skip = None): 

    print(args)
    if not os.path.exists(args.output_path): 
        os.mkdir(args.output_path) 

    best_epoch_path = os.path.join(args.output_path, 'best_epoch')
    if not os.path.exists(best_epoch_path):
        os.mkdir(best_epoch_path)

    b = time.time() 
    val_results = []

    cudnn.benchmark = True
    user_num, item_num, train_data_len = load_all_custom( data_path = data_path)
    print("LOADED DATASET WITH {} users, {} items".format(user_num, item_num))

    raw_train_data = pd.read_csv(f'{data_path}/train_df')    
    val_data_without_neg = pd.read_csv(f'{data_path}/val_df')    
    val_data_with_neg = pd.read_csv(f'{data_path}/val_df_with_neg')    
    test_data_without_neg = pd.read_csv(f'{data_path}/test_df')    
    test_data_with_neg = pd.read_csv(f'{data_path}/test_df_with_neg')    
    sid_pop_total = pd.read_csv(f'{data_path}/sid_pop_total')
    sid_pop_train = pd.read_csv(f'{data_path}/sid_pop_train')

    plain_adj = sp.load_npz(f'{data_path}/s_adj_mat.npz')
    norm_adj = sp.load_npz(f'{data_path}/s_norm_adj_mat.npz')
    mean_adj = sp.load_npz(f'{data_path}/s_mean_adj_mat.npz')

    print('step 1 done')


    train_dataset = BPRData(train_data_len*args.num_ng, data_path)
    train_loader = data.DataLoader(
        train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.model == 'NGCF':
        if args.sample == 'macr':
            model = NGCF_MACR(user_num, item_num, norm_adj).cuda() 
        else:
            model = NGCF(args, user_num, item_num, norm_adj).cuda() 
    if args.model == 'LightGCN':
        if args.sample == 'macr':
            model = LightGCN_MACR(user_num, item_num, norm_adj).to(torch.device('cuda:' + str(0)))
        else:
            model = LightGCN(user_num, item_num, norm_adj).to(torch.device('cuda:' + str(0)))

    if pretrained_path: 
        print("Loading trained model from epoch{}".format(epoch_skip))
        model = torch.load(pretrained_path).cuda() 
        print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('step 2 done')

    sid_pop_train_dict = dict(list(zip(sid_pop_train.sid, sid_pop_train.train_counts)))

    print(args.dataset, ' ', args.model, ' ', args.sample, ' ', args.weight, ' ', 'reg', args.reg, 'burnin', args.burnin)

    print('entered training')

    count, best_hr = 0, 0

    sample = args.sample
    acc_w = 1-args.weight
    pop_w = args.weight

    start = 0 
    if epoch_skip != None: 
        start = epoch_skip
    for epoch in tqdm(range(start, args.epochs)):
        print('epoch is : ',epoch)        
        model.train() 
        start_time = time.time()
        
        print("Loading data")
        train_loader.dataset.get_data(args.dataset, epoch)
        model.cuda()
        print("data loaded")

        if epoch < args.epochs/4:
            if args.burnin == 'yes':
                sample = 'none'            
        elif epoch >= args.epochs/4:
            sample = args.sample        
        
        
        if sample in ['none', 'posneg']:
            print("sampling")
            for user, pos1, pos2, neg1, neg2 in train_loader:    
                pos, neg = pos1, neg1
                _, _ = pos2, neg2                                    
                
                user = user.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

                model.zero_grad()

                u_emb, pos_emb, neg_emb = model(user, pos, neg, drop_flag = True)
                pos_scores = torch.sum(torch.mul(u_emb, pos_emb), axis=1)
                neg_scores = torch.sum(torch.mul(u_emb, neg_emb), axis=1)
                if sample == 'none':
                    loss = - (pos_scores - neg_scores).sigmoid().log().mean()
                elif sample == 'posneg':
                    acc_loss = - (pos_scores - neg_scores).sigmoid().log().mean()/2                
                    pop_loss =  -(1 -(pos_scores + neg_scores).abs() .tanh() ).log().mean()/2
                    loss = acc_loss*acc_w + pop_loss*pop_w
                if args.reg == 'yes':
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos_emb_w = model.embedding_dict.item_emb[pos]
                    neg_emb_w = model.embedding_dict.item_emb[neg]                
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/3 / args.batch_size
                    loss += 1e-5*reg        
                loss.backward()
                optimizer.step()
                
                
                
        elif sample in ['pd']:
            for user, pos1, pos2, neg1, neg2 in tqdm(train_loader):    
                pos, neg = pos1, neg1
                _, _ = pos2, neg2                                    
                
                user = user.cuda()
                pos = pos.cuda()
                neg = neg.cuda()
                model.zero_grad()
                            
                pos1_label = pos1.cpu().tolist()
                neg1_label = neg1.cpu().tolist()
                pos1_map = [sid_pop_train_dict[key] for key in pos1_label]
                neg1_map = [sid_pop_train_dict[key] for key in neg1_label]
                
                pos1_weight = torch.from_numpy(np.array(pos1_map)).cuda()            
                neg1_weight = torch.from_numpy(np.array(neg1_map)).cuda()            
                
                u_emb, pos_emb, neg_emb = model(user, pos, neg, drop_flag = True)
                pos_scores = torch.sum(torch.mul(u_emb, pos_emb), axis=1)
                neg_scores = torch.sum(torch.mul(u_emb, neg_emb), axis=1)
                m = nn.ELU()
                loss = - ( ( m(pos_scores) +1.0 )*pos1_weight**(acc_w) - (  m(neg_scores)+1.0 )*neg1_weight**(acc_w) ).sigmoid().log().mean()
                    
                
                if args.reg == 'yes':
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos_emb_w = model.embedding_dict.item_emb[pos]
                    neg_emb_w = model.embedding_dict.item_emb[neg]                
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/3 / args.batch_size
                    loss += 1e-5*reg        
                loss.backward()
                optimizer.step()            
                
                
        elif sample == 'pos2':
            for user, pos1, pos2, neg1, neg2 in tqdm(train_loader):    
                _ = neg2
                
                user = user.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                neg = neg1.cuda()

                model.zero_grad()
                
                u_emb, pos_1_emb, neg_emb = model(user, pos1, neg, drop_flag = True)            
                u_emb, pos_2_emb, neg_emb = model(user, pos2, neg, drop_flag = True)
                
                pos1_scores = torch.sum(torch.mul(u_emb, pos_1_emb), axis=1)
                pos2_scores = torch.sum(torch.mul(u_emb, pos_2_emb), axis=1)
                neg_scores = torch.sum(torch.mul(u_emb, neg_emb), axis=1)
                
                acc_loss = - (pos1_scores - neg_scores).sigmoid().log().mean()/4 - (pos2_scores - neg_scores).sigmoid().log().mean()/4            
                pop_loss =  -(1 -(pos1_scores - pos2_scores).abs() .tanh() ).log().mean()/2            
                loss = acc_loss*acc_w + pop_loss*pop_w
                
                if args.reg == 'yes':            
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos1_emb_w = model.embedding_dict.item_emb[pos1]
                    pos2_emb_w = model.embedding_dict.item_emb[pos2]
                    neg_emb_w = model.embedding_dict.item_emb[neg]            
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/4 / args.batch_size
                    loss += 1e-5*reg        
                loss.backward()
                optimizer.step()         

        elif sample == 'neg2':
            for user, pos1, pos2, neg1, neg2 in tqdm(train_loader):    
                _ = pos2            
                
                user = user.cuda()
                pos = pos1.cuda()
                #pos2 = pos2.cuda()
                neg1 = neg1.cuda()
                neg2 = neg2.cuda()            

                model.zero_grad()
                
                u_emb, pos_emb, neg_1_emb = model(user, pos, neg1, drop_flag = True)            
                u_emb, pos_emb, neg_2_emb = model(user, pos, neg2, drop_flag = True)
                
                pos_scores = torch.sum(torch.mul(u_emb, pos_emb), axis=1)
                #pos2_scores = torch.sum(torch.mul(u_emb, pos_2_emb), axis=1)
                neg1_scores = torch.sum(torch.mul(u_emb, neg_1_emb), axis=1)
                neg2_scores = torch.sum(torch.mul(u_emb, neg_2_emb), axis=1)            
                
                acc_loss = - (pos_scores - neg1_scores).sigmoid().log().mean()/4 - (pos_scores - neg2_scores).sigmoid().log().mean()/4            
                #loss +=  -(1 -(pos1_scores - pos2_scores).abs() .sigmoid() ).log().mean()/4             
                pop_loss =  -(1 -(neg1_scores - neg2_scores).abs() .tanh() ).log().mean()/2         
                loss = acc_loss*acc_w + pop_loss*pop_w           
                
                if args.reg == 'yes':                        
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos_emb_w = model.embedding_dict.item_emb[pos]
                    #pos2_emb_w = model.embedding_dict.item_emb[pos2]
                    neg1_emb_w = model.embedding_dict.item_emb[neg1]            
                    neg2_emb_w = model.embedding_dict.item_emb[neg2]                        
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2  + torch.norm(neg2_emb_w) ** 2)/4 / args.batch_size
                    loss += 1e-5*reg        
                loss.backward()
                optimizer.step()                 
                
        if sample == 'pos2neg2':
            for user, pos1, pos2, neg1, neg2 in tqdm(train_loader):
                user = user.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                neg1 = neg1.cuda()
                neg2 = neg2.cuda()            

                model.zero_grad()
                
                u_emb, pos_1_emb, neg_1_emb = model(user, pos1, neg1, drop_flag = True)            
                u_emb, pos_2_emb, neg_2_emb = model(user, pos2, neg2, drop_flag = True)
                
                pos1_scores = torch.sum(torch.mul(u_emb, pos_1_emb), axis=1)
                pos2_scores = torch.sum(torch.mul(u_emb, pos_2_emb), axis=1)
                neg1_scores = torch.sum(torch.mul(u_emb, neg_1_emb), axis=1)
                neg2_scores = torch.sum(torch.mul(u_emb, neg_2_emb), axis=1)            
                
                acc_loss = - (pos1_scores - neg1_scores).sigmoid().log().mean()/4 - (pos2_scores - neg2_scores).sigmoid().log().mean()/4            
                pop_loss =  -(1 -(pos1_scores - pos2_scores).abs() .tanh() ).log().mean()/4 -(1 -(neg1_scores - neg2_scores).abs() .tanh() ).log().mean()/4                                    
                loss = acc_loss*acc_w + pop_loss*pop_w           
                
                if args.reg == 'yes':            
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos1_emb_w = model.embedding_dict.item_emb[pos1]
                    pos2_emb_w = model.embedding_dict.item_emb[pos2]
                    neg1_emb_w = model.embedding_dict.item_emb[neg1]            
                    neg2_emb_w = model.embedding_dict.item_emb[neg2]                                    
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2  + torch.norm(neg2_emb_w) ** 2)/5 / args.batch_size
                    loss += 1e-5*reg        
                
                loss.backward()
                optimizer.step()   
                
        if sample == 'macr':
            for user, pos1, pos2, neg1, neg2 in train_loader:
                
                user = user.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                neg1 = neg1.cuda()
                neg2 = neg2.cuda()            

                model.zero_grad()
                
                u_emb, pos_1_emb, neg_1_emb = model(user, pos1, neg1, drop_flag = True)            
                u_emb, pos_2_emb, neg_2_emb = model(user, pos2, neg2, drop_flag = True)
                
                pos1_scores = torch.sum(torch.mul(u_emb, pos_1_emb), axis=1)
                pos2_scores = torch.sum(torch.mul(u_emb, pos_2_emb), axis=1)
                neg1_scores = torch.sum(torch.mul(u_emb, neg_1_emb), axis=1)
                neg2_scores = torch.sum(torch.mul(u_emb, neg_2_emb), axis=1)            
                
                loss1 =  - (pos1_scores.sigmoid()*model.macr_user(user).sigmoid()*model.macr_item(pos1).sigmoid()).log().mean()/4 
                - (pos2_scores.sigmoid()*model.macr_user(user).sigmoid()*model.macr_item(pos2).sigmoid()).log().mean()/4 
                - (1-(neg1_scores.sigmoid()*model.macr_user(user).sigmoid()*model.macr_item(neg1).sigmoid())).log().mean()/4
                - (1-(neg2_scores.sigmoid()*model.macr_user(user).sigmoid()*model.macr_item(neg2).sigmoid())).log().mean()/4
                
                loss2 =  - (model.macr_item(pos1).sigmoid()).log().mean()/4 
                - (model.macr_item(pos2).sigmoid()).log().mean()/4 
                - (1- model.macr_item(neg1).sigmoid() ).log().mean()/4
                - (1- model.macr_item(neg2).sigmoid() ).log().mean()/4            
                
                loss3 =  - (model.macr_user(user).sigmoid()).log().mean()/4 
                - (model.macr_user(user).sigmoid()).log().mean()/4 
                - (1- model.macr_user(user).sigmoid() ).log().mean()/4
                - (1- model.macr_user(user).sigmoid() ).log().mean()/4            
                
                loss = loss1 + 0.0005*loss2 + 0.0005*loss3
                
                if args.reg == 'yes':            
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos1_emb_w = model.embedding_dict.item_emb[pos1]
                    pos2_emb_w = model.embedding_dict.item_emb[pos2]
                    neg1_emb_w = model.embedding_dict.item_emb[neg1]            
                    neg2_emb_w = model.embedding_dict.item_emb[neg2]                                    
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2  + torch.norm(neg2_emb_w) ** 2)/5 / args.batch_size
                    loss += 1e-5*reg        
                
                loss.backward()
                optimizer.step()              
                
                
        if sample == 'ipw':
            for user, pos1, pos2, neg1, neg2 in train_loader:
                
                user = user.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                neg1 = neg1.cuda()
                neg2 = neg2.cuda()            

                model.zero_grad()
                
                u_emb, pos_1_emb, neg_1_emb = model(user, pos1, neg1, drop_flag = True)            
                u_emb, pos_2_emb, neg_2_emb = model(user, pos2, neg2, drop_flag = True)
                
                pos1_scores = torch.sum(torch.mul(u_emb, pos_1_emb), axis=1)
                pos2_scores = torch.sum(torch.mul(u_emb, pos_2_emb), axis=1)
                neg1_scores = torch.sum(torch.mul(u_emb, neg_1_emb), axis=1)
                neg2_scores = torch.sum(torch.mul(u_emb, neg_2_emb), axis=1)            
                
                pos1_label = pos1.cpu().tolist()
                pos2_label = pos2.cpu().tolist()           
                neg1_label = neg1.cpu().tolist()
                neg2_label = neg2.cpu().tolist()
                pos1_map = [sid_pop_train_dict[key] for key in pos1_label]
                pos2_map = [sid_pop_train_dict[key] for key in pos2_label]            
                neg1_map = [sid_pop_train_dict[key] for key in neg1_label]
                neg2_map = [sid_pop_train_dict[key] for key in neg2_label]            
                
                pos1_weight = torch.from_numpy(1/np.array(pos1_map)).cuda()            
                pos2_weight = torch.from_numpy(1/np.array(pos2_map)).cuda()                        
                neg1_weight = torch.from_numpy(1/np.array(neg1_map)).cuda()            
                neg2_weight = torch.from_numpy(1/np.array(neg2_map)).cuda()                        
                
                pop_loss =  - (pos1_weight*(pos1_scores).sigmoid().log()).mean()/4 
                - (pos2_weight*(pos2_scores).sigmoid().log()).mean()/4 
                - (neg1_weight*(1-(neg1_scores).sigmoid()).log()).mean()/4 
                - (neg2_weight*(1-(neg2_scores).sigmoid()).log()).mean()/4
                
                
                
                loss = pop_loss
                
                if args.reg == 'yes':            
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos1_emb_w = model.embedding_dict.item_emb[pos1]
                    pos2_emb_w = model.embedding_dict.item_emb[pos2]
                    neg1_emb_w = model.embedding_dict.item_emb[neg1]            
                    neg2_emb_w = model.embedding_dict.item_emb[neg2]                                    
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2  + torch.norm(neg2_emb_w) ** 2)/5 / args.batch_size
                    loss += 1e-5*reg        
                
                loss.backward()
                optimizer.step()     
                
                
        elif sample == 'pearson':
            for user, pos1, pos2, neg1, neg2 in train_loader:    
                pos, neg = pos1, neg1
                _, _ = pos2, neg2            
                
                user = user.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

                model.zero_grad()

                u_emb, pos_emb, neg_emb = model(user, pos, neg, drop_flag = True)
                pos_scores = torch.sum(torch.mul(u_emb, pos_emb), axis=1)
                neg_scores = torch.sum(torch.mul(u_emb, neg_emb), axis=1)
                loss = - (pos_scores - neg_scores).sigmoid().log().mean()
                
                
                if args.reg == 'yes':                        
                    user_emb_w = model.embedding_dict.user_emb[user]        
                    pos_emb_w = model.embedding_dict.item_emb[pos]
                    neg_emb_w = model.embedding_dict.item_emb[neg]                            
                    reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/3 / args.batch_size
                    loss += 1e-5*reg
                
                loss.backward()
                optimizer.step()    
                
            model.zero_grad()           
            pcc = pcc_train(model, raw_train_data, sid_pop_train, item_num)            
            loss = acc_w*(pcc**2)
            loss.backward()
            optimizer.step()            
        model.eval()
        print('entered evaluated')
        
            
        HR, NDCG, ARP = metrics_graph_bpr(model, val_data_with_neg, args.top_k, sid_pop_total, user_num)
        #HR, NDCG, ARP = 0, 0, 0
        PCC_TEST = pcc_test(model, val_data_without_neg, sid_pop_total, item_num)   
        #PCC_TEST2 = pcc_test_check(model, val_data_without_neg, sid_pop_total)
        
        score = pred_item_score(model, val_data_without_neg, sid_pop_total)
        SCC_score_test = stats.spearmanr(score.dropna()['sid_pop_count'].values, score.dropna()['pred'].values)   
        rank = pred_item_rank(model, val_data_without_neg, sid_pop_total)    
        SCC_rank_test = stats.spearmanr(rank.dropna()['sid_pop_count'].values, rank.dropna()['rank'].values)
        
        upo = uPO(model, val_data_without_neg, sid_pop_total)    
        
        rankdist = pred_item_rankdist(model, val_data_without_neg, sid_pop_total)
        mean_test = np.mean(rankdist[rankdist.notna()].values)    
        skew_test = skew(rankdist[rankdist.notna()].values)

        model.eval()    
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}\tARP: {:.3f}".format(np.mean(HR), np.mean(NDCG), np.mean(ARP)))

        print('PCC_TEST : ', np.round(PCC_TEST, 3))   
        #print('PCC_TEST check : ', np.round(PCC_TEST2, 3))           
        print('SCC_score_test : ', np.round(SCC_score_test[0], 3))        
        print('SCC_rank_test : ', np.round(SCC_rank_test[0], 3))
        print('upo is :', np.round(upo, 3))            
        print('mean_test : ', np.round(mean_test, 3))        
        print('skew_test : ', np.round(skew_test, 3))        
        print(' ')    
        epoch_val_result = [args.batch_size, epoch, args.sample, args.weight, HR, NDCG, ARP, PCC_TEST, SCC_score_test[0], SCC_rank_test[0],np.round(upo, 3), mean_test, skew_test]
        val_results.append(epoch_val_result)

        if HR > best_hr:
            best_hr, best_ndcg, best_arp, best_epoch = HR, NDCG, ARP, epoch 
            backup_fpath = os.path.join(best_epoch_path, "model{}.pth".format(epoch))
            torch.save(model, backup_fpath)

        if args.out:
            checkpoint_path = os.path.join(args.output_path, "checkpoints")
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(model, 
                '{}/model{}.pth'.format(checkpoint_path, epoch))
       
    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, ARP = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg, best_arp))

    model.cuda()
    model.eval()
    print('entered test evaluated')    
    'HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)'
    HR, NDCG, ARP = metrics_graph_bpr(model, test_data_with_neg, args.top_k, sid_pop_total, user_num)
    #HR, NDCG, ARP = 0, 0, 0

    PCC_TEST = pcc_test(model, test_data_without_neg, sid_pop_total, item_num)  
    
    score = pred_item_score(model, test_data_without_neg, sid_pop_total)
    SCC_score_test = stats.spearmanr(score.dropna()['sid_pop_count'].values, score.dropna()['pred'].values)    
    rank = pred_item_rank(model, test_data_without_neg, sid_pop_total)    
    SCC_rank_test = stats.spearmanr(rank.dropna()['sid_pop_count'].values, rank.dropna()['rank'].values)

    upo = uPO(model, test_data_without_neg, sid_pop_total)

    rankdist = pred_item_rankdist(model, test_data_without_neg, sid_pop_total)
    mean_test = np.mean(rankdist[rankdist.notna()].values)    
    skew_test = skew(rankdist[rankdist.notna()].values)

    epoch_val_result = [args.batch_size, -1, args.sample, args.weight, HR, NDCG, ARP, PCC_TEST, SCC_score_test[0], SCC_rank_test[0], np.round(upo,3), mean_test, skew_test]
    val_results.append(epoch_val_result)

    experiment_results = pd.DataFrame(val_results)
    experiment_results.columns = ['batch', 'epoch', 'sample', 'weight', 'HR', 'NDCG', 'ARP', 'PCC', 'SCC_score', 'SCC_rank',  'upo','mean', 'skew']

     

    elapsed_time = time.time() - start_time
    print(args.dataset, ' ', args.model, ' ', args.sample, ' ', args.weight, ' ', 'reg', args.reg, 'burnin', args.burnin)
    print(' ')
    print("HR: {:.3f}\tNDCG: {:.3f}\tARP: {:.3f}".format(np.mean(HR), np.mean(NDCG), np.mean(ARP)))
    print('PCC_TEST : ', np.round(PCC_TEST, 3))    
    print('SCC_score_test : ', np.round(SCC_score_test[0], 3))        
    print('SCC_rank_test : ', np.round(SCC_rank_test[0], 3))  
    print('upo is :', np.round(upo, 3)) 
    print('mean_test : ', np.round(mean_test, 3))        
    print('skew_test : ', np.round(skew_test, 3))        
    print(' ')    

    a = time.time()

    print("PROCESS TOOK:{}".format(a-b))
    torch.save(model, args.output_path+"model.pt")
    print("Saved model to {}".format(args.output_path+"model.pt"))

def validation_c_MACR(cfg, model_path, output_path, data_path, model_name, rec_num, c = None): 
    user_num, item_num, train_data_len = load_all_custom(data_path)
    plain_adj = sp.load_npz(f'{data_path}/s_adj_mat.npz')
    norm_adj = sp.load_npz(f'{data_path}/s_norm_adj_mat.npz')
    mean_adj = sp.load_npz(f'{data_path}/s_mean_adj_mat.npz')

    test_data = pd.read_csv(f'{data_path}/val_df') 
    total_data =  pd.read_csv(f'{data_path}/total_df') 

    test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
    test_data['sid'] = test_data['sid'].apply(lambda x : int(x))    
    test_users_num = len(test_data['uid'].unique())  
    test_users = test_data.uid.unique()  

    total_data['sid'] = total_data['sid'].apply(lambda x : int(x))  

    user = test_users
    user = np.array(user).astype(np.int32)
    user = user.tolist()
    user = torch.LongTensor(user).cuda()

    item = total_data['sid'].unique() #.tolist()
    item = np.array(item).astype(np.int32)        
    item = item.tolist()    
    item = torch.LongTensor(item).cuda()


    model = LightGCN_MACR(user_num, item_num, norm_adj)

    model = torch.load(model_path)
    item_emb = model.embedding_dict.item_emb
    user_emb = model.embedding_dict.user_emb[test_data['uid'].unique()]
    item = item.cpu()
    user = user.cpu() 
    user_ego = (model.macr_user(user) * user_emb).sigmoid()
    item_ego = (model.macr_item(item)* item_emb).sigmoid() 
    ui_scores = user_emb@item_emb.T - c * user_ego@item_ego.T
    values, idx = torch.topk(ui_scores, rec_num)
    rec_df = pd.DataFrame({'pid': test_data['uid'].unique(), 'recs': idx.tolist()})

    if output_path: 
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,f"rec_tracks.pkl")
        print("***Saving Recommended Track List to {}***".format(file_path))
        pickle.dump(rec_df, open(file_path, "wb"))
    return rec_df, test_data


def make_recs_OG(cfg, model_path, output_path, data_path, model_name, rec_num, c = None, mode='test'): 
    print(f'model:{model_name}, mode:{mode}')
    # data_path = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/src2/benchmarks/ScoreReg'
    # data_path='/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/ScoreReg/10K_PID_V2'

    user_num, item_num, train_data_len = load_all_custom(data_path)
    plain_adj = sp.load_npz(f'{data_path}/s_adj_mat.npz')
    norm_adj = sp.load_npz(f'{data_path}/s_norm_adj_mat.npz')
    mean_adj = sp.load_npz(f'{data_path}/s_mean_adj_mat.npz')

    if mode == 'valid': 
        test_data = pd.read_csv(f'{data_path}/val_df') 
        test_data['pid_for_recs'] = test_data['uid']
        print(test_data)
    else: 
        test_data = pd.read_csv(f'{data_path}/test_df') 
    total_data =  pd.read_csv(f'{data_path}/total_df') 

    test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
    test_data['sid'] = test_data['sid'].apply(lambda x : int(x))    
    test_users_num = len(test_data['uid'].unique())  
    test_users = test_data.uid.unique()  

    total_data['sid'] = total_data['sid'].apply(lambda x : int(x))  

    user = test_users
    user = np.array(user).astype(np.int32)
    user = user.tolist()
    user = torch.LongTensor(user).cuda()

    item = total_data['sid'].unique() #.tolist()
    item = np.array(item).astype(np.int32)        
    item = item.tolist()    
    item = torch.LongTensor(item).cuda()

    if model_name == "LightGCN": 
        model = LightGCN(user_num, item_num, norm_adj) #.cuda()
    if model_name == 'MACR': 
        model = LightGCN_MACR(user_num, item_num, norm_adj)
    if model_name == "NGCF": 
        model = NGCF(user_num, item_num, norm_adj)

    model = torch.load(model_path).cuda() 
    # u_emb, pos_i_emb, neg_i_emb = model(user, item, item, drop_flag = False)
    # predictions = user_emb@item_emb.T #torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
    # test_data['pred'] = predictions.detach().cpu()

    
    # test_data['topk'] = test_data['pred'].apply(lambda x: torch.topk(x, rec_num)) 
    # rec_df = pd.DataFrame({'pid': test_data['pid_for_recs'], 'recs': test_data['pred']}) 
    
   
    if model_name == 'MACR':
        model = torch.load(model_path)
        item_emb = model.embedding_dict.item_emb
        user_emb = model.embedding_dict.user_emb[test_data['uid'].unique()]
        item = item.cpu()
        user = user.cpu() 
        user_ego = (model.macr_user(user) * user_emb).sigmoid()
        item_ego = (model.macr_item(item)* item_emb).sigmoid() 
        print(item_ego, user_ego)
        ui_scores = user_emb@item_emb.T - c * user_ego@item_ego.T
       
        
    else: 
        item_emb = model.embedding_dict.item_emb.cpu()
        user_emb = model.embedding_dict.user_emb.cpu()[test_data['uid'].unique()]
        ui_scores = user_emb@item_emb.T
    values, idx = torch.topk(ui_scores, rec_num)

    # test_pids = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, "rb"))['pid'].unique()

    rec_df = pd.DataFrame({'pid': test_data['pid_for_recs'].unique(), 'recs': idx.tolist()})

    file_path = output_path
    if output_path: 
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        if c != None: 
            file_path = os.path.join(output_path,f"rec_tracks_{c}.pkl")
        else: 
            file_path = os.path.join(output_path,"rec_tracks.pkl")
        print("***Saving Recommended Track List to {}***".format(file_path))
        pickle.dump(rec_df, open(file_path, "wb"))
    return rec_df, file_path
    
def save_item_emb(model_path, output_path, data_path, model_name): 
    # import ipdb 
    # ipdb.set_trace() 
    # data_path = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/src2/benchmarks/ScoreReg'
    # data_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/ScoreReg/data/final_mus'
    user_num, item_num, train_data_len = load_all_custom(data_path)
    
    c = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/LFM/subset/Score_Reg/v1_MACR/checkpoints/model18.pth'
    plain_adj = sp.load_npz(f'{data_path}/s_adj_mat.npz')
    norm_adj = sp.load_npz(f'{data_path}/s_norm_adj_mat.npz')
    mean_adj = sp.load_npz(f'{data_path}/s_mean_adj_mat.npz')

    # if model_name == "MACR": 
    #     model = NGCF_MACR(user_num, item_num, norm_adj).cuda()
    #     model = torch.load(model_path)
    #     item_emb = model.embedding_dict.item_emb.detach().numpy() 
    #     macr_emb = model.macr_item
    #     macr_item_emb = []
    #     for idx in range(len(item_emb)): 
    #         new_emb = item_emb
    #     pickle.dump(item_emb, open(output_path, "wb"))
    
    if model_name == "LightGCN": 
        model = LightGCN(user_num, item_num, norm_adj) #.cuda()
    if model_name == "NGCF": 
        model = NGCF("", user_num, item_num, norm_adj).cuda()
    model = torch.load(model_path)
    item_emb = model.embedding_dict.item_emb.detach().numpy() 
    pickle.dump(item_emb, open(output_path, "wb"))
    