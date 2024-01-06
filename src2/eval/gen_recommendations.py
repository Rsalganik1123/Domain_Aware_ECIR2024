from src2.utils.config.small import get_cfg_defaults
from src2.model.build import build_model
from src2.graph_build.data_load import build_dataset
import src2.graph_build.spotify_dataset
from torch.utils.data import IterableDataset, DataLoader
from src2.sampler.graph_sampler import build_graph_sampler
# from tqdm import tqdm 
from torch.nn.functional import cosine_similarity
from torch.nn import CosineSimilarity
import torch 
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm 
import time 
import numpy as np 
from numpy.linalg import norm 
import os 
# import ipdb

'''
File containing functions for generating recommendations from PinSAGE track embeddings
'''


def simi(output):  # new_version
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)
    # print("similarity matrix shape", res.shape)
    return res


def sim(playlist_embedding, track_embeddings, k, version):
    if version == 'cosine': 
        r = torch.Tensor(cosine_similarity(playlist_embedding, track_embeddings))
        return r 
    if version == 'dot_prod': 
        r = torch.mm(playlist_embedding, track_embeddings.T).squeeze()
        return r
    if version == 'scaled_cosine': 
        track_norm = track_norm/norm(track_embeddings, axis=1)
        playlist_norm = playlist_norm/norm(playlist_embedding)
        r = 5 * (torch.mm(track_norm, playlist_norm.transpose(0, 1)) + 1)
        return r 
def find_overlaps(gt_tids, gen_amount): 
    all_options = []
    for t in gt_tids: 
        r = torch.Tensor(cosine_similarity(track_embeddings[t], track_embeddings))
        recs = torch.topk(r, gen_amount)[1].tolist() 
        all_options.append(recs)
    print(Counter(all_options))

def gen_recommendations_cosine(cfg, k, gen_amount, sim_version='cosine', track_embeddings=None, track_embed_path=None, output_path=None): 
    '''
    INPUT: 
        cfg: dictionary with all parameters for general PinSAGE run
        k: number of tracks to recommend per playlist 
        gen_amount: number of tracks to use for generating playlist embedding 
        track_embeddings: numpy array containing all track embeddings 
        track_embed_path: path to saved array containing all track embeddings 
        output_path: file path to save recommendations 
    OUTPUT:
        track_recs: tids of all recommended tracks 
        rec_df: dataframe containing recommendations --> columns: 'pid', 'recs' (tids of recommended tracks)
    '''
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))
    print("***generating recommendations***")
    if track_embed_path: 
        track_embeddings = pickle.load(open(track_embed_path, "rb")) 
    track_embeddings = torch.tensor(track_embeddings)
    print("***loaded track embedding of size:{}***".format(track_embeddings.size()))
    recommended_tracks, rec_scores = [] , [] 
    for pid in tqdm(test_set.pid.unique()): 
        associated_tracks = test_set[test_set.pid == pid]['tid'].tolist()
        associated_tracks = associated_tracks[:gen_amount]
        playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
        sim_values = sim(playlist_embedding, track_embeddings, k, sim_version)
        recs = torch.topk(sim_values, k)[1].tolist() 
        scores = sim_values[recs]
        rec_tracks = df_items[df_items.tid.isin(recs)].tid.tolist() 
        
        assert set(recs) == set(rec_tracks)
        recommended_tracks.append(recs)  
        rec_scores.append(scores)
    rec_df = pd.DataFrame({'pid': test_set.pid.unique(), 'recs': recommended_tracks, 'scores': rec_scores})
    # ipdb.set_trace()
    if output_path: 
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,"{}_{}_recommended_tracks.pkl".format(sim_version, gen_amount))
        print("***Saving Recommended Track List to {}***".format(file_path))
        pickle.dump(rec_df, open(file_path, "wb"))
    return rec_df

def gen_recommendations_random(cfg, k, gen_amount, track_embed_path=None, output_path=None): 
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))

    print("***generating recommendations***")
    if track_embed_path: 
        track_embeddings = pickle.load(open(track_embed_path, "rb")) 
    track_embeddings = torch.tensor(track_embeddings)
    print("***loaded track embedding of size:{}***".format(track_embeddings.size()))
    recommended_tracks, rec_scores = [] , [] 
    for pid in tqdm(test_set.pid.unique()): 
        associated_tracks = test_set[test_set.pid == pid]['tid'].tolist()
        associated_tracks = associated_tracks[:gen_amount]
        playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
        recs = np.random.choice(df_items.tid.unique(), k).tolist()
        recommended_tracks.append(recs)  
        rec_scores.append(scores)
    rec_df = pd.DataFrame({'pid': test_set.pid.unique(), 'recs': recommended_tracks})
    if output_path: 
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,"u_rec_tracks.pkl")
        print("***Saving Recommended Track List to {}***".format(file_path))
        pickle.dump(rec_df, open(file_path, "wb"))
    return rec_df, file_path

def gen_recommendations_cosine_clean(cfg, k, gen_amount, track_embed_path=None, output_path=None, mode = 'test'): 
    '''
    INPUT: 
        cfg: dictionary with all parameters for general PinSAGE run
        k: number of tracks to recommend per playlist 
        gen_amount: number of tracks to use for generating playlist embedding 
        track_embeddings: numpy array containing all track embeddings 
        track_embed_path: path to saved array containing all track embeddings 
        output_path: file path to save recommendations 
    OUTPUT:
        track_recs: tids of all recommended tracks 
        rec_df: dataframe containing recommendations --> columns: 'pid', 'recs' (tids of recommended tracks)
    '''
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    
    if mode == 'test':
        print('test mode') 
        test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))
    else: 
        print('val mode')
        valid_idx = all_data['val_indices']
        val_data = df_interactions.loc[valid_idx]
        val_pid = val_data.groupby('pid')['tid'].apply(list).reset_index(name='tids')
        val_pid['num'] = val_pid['tids'].apply(len)
        val_100 = val_pid[val_pid.num >= 100]
        val_100['tids'] = val_100['tids'].apply(lambda x: x[:100])
        val_100['num'] = val_100['tids'].apply(len)
        val_100 = val_100.explode('tids').drop(columns=['num'])
        test_set = val_100.rename(columns={'tids':'tid'})
    print("***generating recommendations***")
    if track_embed_path: 
        track_embeddings = pickle.load(open(track_embed_path, "rb")) 
    track_embeddings = torch.tensor(track_embeddings)
    print("***loaded track embedding of size:{}***".format(track_embeddings.size()))
    recommended_tracks, rec_scores = [] , [] 
    for pid in tqdm(test_set.pid.unique()): 
        associated_tracks = test_set[test_set.pid == pid]['tid'].tolist()
        associated_tracks = associated_tracks[:gen_amount]
        playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
        sim_values = sim(playlist_embedding, track_embeddings, k, 'cosine')
        recs = torch.topk(sim_values, k)[1].tolist() 
        scores = sim_values[recs]
        rec_tracks = df_items[df_items.tid.isin(recs)].tid.tolist() 
        assert set(recs) == set(rec_tracks)
        recommended_tracks.append(recs)  
        rec_scores.append(scores)
    rec_df = pd.DataFrame({'pid': test_set.pid.unique(), 'recs': recommended_tracks, 'scores': rec_scores})
    if output_path: 
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,"u_rec_tracks.pkl")
        print("***Saving Recommended Track List to {}***".format(file_path))
        pickle.dump(rec_df, open(file_path, "wb"))
    return rec_df, file_path