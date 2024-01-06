from tqdm import tqdm 
import pickle 
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# from src2.utils.config.small import get_cfg_defaults
from torch.nn.functional import cosine_similarity, one_hot
import numpy as np 

'''
File contains everything necessary to generate benchmark embeddings using the intital feature set. 
Groups of features supported: 
    1. Title - uses the BERT embeddings of a track's name 
    2. Music - uses the binned (or real number valued) music features scraped from the Spotify API 
        2.a. Currently supporting: danceability, energy, loudness, speechiness, acousticness, instrumenatalness, liveness, valence, tempo 
    3. Artist - uses artist metadata scraped from Spotify API 
        3.a Currently supporting: follower count, popularity 
    4. Image - uses the ResNet50 embeddings for album artworks 
    5. Genre - uses a one hot encoding to indicate all genre tags present for an artist 
'''
def bin_features(data, columns, bin_count):
    '''
    INPUT: 
        data: dataframe containing tracks and all the information related to them. 
        columns: column subset to generate features --> should be used in feature type grouping configuration 
        bin_count: number of bins to use 
    OUTPUT: 
        binned features --> track embeddings 
    '''
    embs = [] 
    for c in columns: 
        embs.append(list(pd.cut(data[c], bin_count, labels=range(bin_count))))
    return np.vstack(embs).T

def scale_features(data, columns, embeddings=False, scale=True):
    '''
    INPUT: 
        data: dataframe containing tracks and all the information related to them. 
        columns: column subset to generate features --> should be used in feature type grouping configuration 
        embeddings: indicates whether the values in column subset are vector or real values, default = False (real values)
        scale: indicates whether to return scaled or raw feature values 
    OUTPUT: 
        scaled / raw feature group --> track embeddings 
    '''
    scaler = StandardScaler()
    if embeddings:
        embs = data[columns]
        to_array = []
        for r in tqdm(embs.index): 
            emb = embs.iloc[r]
            to_array.append(emb[0])
        embs = np.vstack(to_array).astype(float)
    else: embs = data[columns].values.astype(float)
    if scale: 
        scaled_embs = scaler.fit_transform(embs)
    else: return embs
    return scaled_embs 

def gen_baseline_recommendations_all(cfg, name_embeddings = False, music=False, artist = False, img=False, genre=False, k=10, scale=False, bin=False, gen_amount=10, output_path=None): 
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))
    print("Generating Embeddings from:{}".format(cfg.DATASET.DATA_PATH.split("/")[-1]))
    track_embeddings = []
    if name_embeddings: 
        print("adding track embeddings")
        feat = scale_features(df_items, ['track_name_emb'], embeddings=True) 
        track_embeddings.append(feat)
    if music: 
        print("adding music features")
        # if scale: 
        feat = scale_features(df_items, ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo']) 
        track_embeddings.append(feat)
    if artist: 
        print("adding artist features") 
        if bin: 
            feat = bin_features(df_items, ['popularity', 'followers'], bin_count=10)
        else: 
            feat= scale_features(df_items, ['followers'], scale=scale)
            track_embeddings.append(feat)
            feat = scale_features(df_items,['popularity'], scale=scale)
        track_embeddings.append(feat)
    if img: 
        print("adding album artwork embeddings")
        feat = scale_features(df_items,['img_emb'], embeddings=True, scale=False)
        track_embeddings.append(feat)
    if genre: 
        print("adding genre embeddings")
        feat = scale_features(df_items,['genre'], embeddings=True, scale = False) 
        track_embeddings.append(feat)
        
    track_embeddings = np.concatenate(track_embeddings, axis=1).astype(float)
    track_embeddings = torch.Tensor(track_embeddings)
    print("Generated embedding of size:{}".format(track_embeddings.shape))
    recommended_tracks = [] 
    for pid in tqdm(test_set.pid.unique()): 
        associated_tracks = test_set[test_set.pid == pid]['tid'].tolist()
        associated_tracks = associated_tracks[:gen_amount]
        playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
        # r = torch.tensor(cosine_similarity(playlist_embedding, track_embeddings))
        r = torch.Tensor(cosine_similarity(playlist_embedding, track_embeddings))
        recs = torch.topk(r, k)[1].tolist() 
        recommended_tracks.append(recs)
        # rec_tracks = df_items[df_items.tid.isin(recs)].tid.tolist() 
        # recommended_tracks.append(rec_tracks)
        # recs = (torch.topk(r, k))[1].tolist() 
        # rec_tracks = df_items[df_items.tid.isin(recs)]
        # recommended_tracks.append(rec_tracks.tid.tolist())

    rec_df = pd.DataFrame({'pid': test_set.pid.unique(), 'recs': recommended_tracks})
    if output_path: 
        print("***Saving Recommended Track List to {}***".format(output_path))
        pickle.dump(rec_df, open(output_path, "wb"))
    return rec_df

def gen_baseline_recommendations_small(cfg, name_embeddings = True, music=True, artist = True, img=True, genre=True, bin = False, scale=True, k=10, gen_amount=10, output_path=None): 
    '''
    INPUT 
        cfg: default settings for the entire run 
        name_embeddings: adding BERT embeddings of track names (feature group 1)
        music: adding audio metadata (feature group 2)
        artist: adding artist metadata (feature group 3)
        img: adding ResNet50 embeddings of album artworks (feature group 4)
        genre: adding one hot encoding of genre tags (feature grop 5)
        k: number of recommendations per playlist 
        gen_amount: number of tracks to use in order to generate a playlist
        output_path: where to save the recommendations
    OUTPUT: 
        recommended tracks: list of tids that were recommended 
        rec_df: dataframe containing recommendations --> columns: 'pid', 'recs' (tids of recommended tracks)
    '''
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    test_user_ids = all_data['test_pids']
    print("Generating Embeddings from:{}".format(cfg.DATASET.DATA_PATH.split("/")[-1]))
    track_embeddings = []
    if name_embeddings: 
        print("adding track embeddings")
        feat = scale_features(df_items, ['track_name_emb'], embeddings=True, scale=scale) 
        track_embeddings.append(feat)
    if music: 
        print("adding music features")
        if bin: 
            feat = bin_features(df_items, ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo'], bin_count=10)
        else: 
            feat = scale_features(df_items, ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo'], scale=scale) 
        track_embeddings.append(feat)
    if artist: 
        print("adding artist features") 
        if bin: 
            feat = bin_features(df_items, ['popularity', 'followers'], bin_count=10)
        else: 
            feat= scale_features(df_items, ['followers'], scale=scale)
            track_embeddings.append(feat)
            feat = scale_features(df_items,['popularity'], scale=scale)
        track_embeddings.append(feat)
    if img: 
        print("adding album artwork embeddings")
        feat = scale_features(df_items,['img_emb'], embeddings=True, scale=scale)
        track_embeddings.append(feat)
    if genre: 
        print("adding genre embeddings")
        feat = scale_features(df_items,['genre'], embeddings=True, scale = scale) 
        track_embeddings.append(feat)
        
    track_embeddings = np.concatenate(track_embeddings, axis=1).astype(float)
    track_embeddings = torch.tensor(track_embeddings)
    print("Generated embedding of size:{}".format(track_embeddings.shape))
    recommended_tracks = [] 
    for pid in tqdm(test_user_ids): 
        associated_tracks = df_interactions[df_interactions.pid == pid]['tid'].tolist()
        associated_tracks = associated_tracks[:min(gen_amount, len(associated_tracks))]
        tids_in_graph = df_items[df_items.tid.isin(associated_tracks)].tid_in_graph.tolist()
        playlist_embedding = torch.mean(track_embeddings[tids_in_graph], axis=0).reshape(1, -1)
        r = torch.tensor([cosine_similarity(playlist_embedding, t) for t in track_embeddings])
        recs = (torch.topk(r, k))[1].tolist() 
        rec_tracks = df_items[df_items.tid_in_graph.isin(recs)]
        recommended_tracks.append(rec_tracks.tid.tolist())

    rec_df = pd.DataFrame({'pid': test_user_ids, 'recs': recommended_tracks})
    if output_path: 
        print("***Saving Recommended Track List to {}***".format(output_path))
        pickle.dump(rec_df, open(output_path, "wb"))
    return recommended_tracks, rec_df