from torch.nn.functional import cosine_similarity
import torch 
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm
import os 
from torch.utils.data import IterableDataset, DataLoader


def sim(playlist_embedding, track_embeddings):
    r = torch.Tensor(cosine_similarity(playlist_embedding, track_embeddings))
    return r 

def LT_perc(rec_df, items, LT_col, k, LT = 0): 
    recs = rec_df.explode('recs', ignore_index=True)
    recs['tid'] = recs.recs
    recs = pd.merge(recs, items, on='tid')
    LT_df = recs.groupby('pid')[LT_col].apply(lambda x: len([i for i in x if i <=LT])).reset_index(name='#LT')
    LT_df['%LT'] = LT_df['#LT'].apply(lambda x: x/k)  
    return  np.mean(LT_df['%LT'])

def recall(rec_df, gt, k=20):
    recall_vals = [] 
    for pid in rec_df.pid.unique(): 
        gt_tids = gt[gt.pid == pid].tid.to_list()
        k = min(k, len(gt_tids))
        gt_tids = gt_tids[:k]
        preds = rec_df[rec_df.pid == pid].recs.to_list()[0]
        preds = preds[:len(gt_tids)]
        overlap = [i for i in gt_tids if i in preds] 
        portion = len(overlap) / len(gt_tids)
        recall_vals.append(portion)
    return np.mean(recall_vals)

def rec_validation(cfg, model, val_dataloader, val_data, items, mode='fairness', k=100): 
    print("***Running Validation loop***")
    gen_amount = 10 
    model.eval()
    model = model.cuda()
    device = torch.device('cuda:0')
    #generate embeddings on validation graph 
    all_features_cpu = []
    feat_list = []  
    all_ids = [] 
    dataloader_it = iter(val_dataloader)
    
    for blocks in tqdm(dataloader_it):
        with torch.no_grad():
            for i in range(len(blocks)): 
                blocks[i] = blocks[i].to(device)
            features = model.get_repr(blocks)
            all_features_cpu.append(features.cpu().numpy())
    
    all_features_array = np.concatenate(all_features_cpu)
    track_embeddings = torch.tensor(all_features_array)
    recommended_tracks, rec_scores, pids = [] , [], [] 
    
    for pid in tqdm(val_data.pid.sample(50, replace=False, random_state = 32)): 
        pids.append(pid)
        associated_tracks = val_data[val_data.pid == pid]
        associated_tracks = associated_tracks.tid.tolist()[:gen_amount] 
        playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
        sim_values = sim(playlist_embedding, track_embeddings)
        recs = torch.topk(sim_values, k)[1].tolist() 
        scores = sim_values[recs]
        recommended_tracks.append(recs)  
        rec_scores.append(scores)

    rec_df = pd.DataFrame({'pid': pids, 'recs': recommended_tracks, 'scores': rec_scores})
    
    if mode == 'utility': 
        avg_recall = recall(rec_df, val_data)
        return avg_recall
    if mode == 'fairness': 
        avg_LT = LT_perc(rec_df, items,  cfg.FAIR.POP_FEAT, k)
        return avg_LT 


