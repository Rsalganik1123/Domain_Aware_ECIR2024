import pickle
import os
import re
import json
import numpy as np
import pandas as pd 
import datetime
import tqdm 
import pickle 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from lightfm import LightFM
from collections import defaultdict
import time 

dataset_path = 'FILL IN'
def process_mpd(output_file, prev_songs_window):
    max_prev_song = 0
    previous_tracks = defaultdict(lambda: defaultdict(int))
    playlists_tracks = []
    playlists = []
    playlists_extra = {'name': []}
    
    print("***LOADING Train Playlists***")

    data = pickle.load(open(f'{dataset_path}datasets/small_100_10/CF_trainval.pkl', "rb")) #contig_2
    test = pickle.load(open(f'{dataset_path}/datasets/small_100_10/CF_test.pkl', "rb"))
    
    print("DATA loaded")
    for pid in tqdm.tqdm(data.pid.unique()):
        
        row = data[data.pid == pid]
        pos = row['pos'].tolist()[0]
        tracks = row['track_uri'].tolist()[0]
        sorted_tracks = [x for x, _ in sorted(zip(tracks, pos), key=lambda pair: pair[1])]
        
        tracks = defaultdict(int)
    
        for track in sorted_tracks:
            tracks[track] += 1
        playlists_tracks.append(tracks)
        playlists.append(str(pid))
    
    # Add playlists on testing set
    test_playlists = []
    test_playlists_tracks = []
    
    train_playlists_count = len(playlists)
    test_playlists_recommended_sum = []
    print("***Loading test playlists***")
    for pid in tqdm.tqdm(test.pid.unique()):
        row = test[test.pid == pid]
        
        
        playlists.append(str(pid))
        test_playlists.append(str(pid))
        

        tracks = defaultdict(int)
        for track in row['track_uri'].tolist()[0]: #track_uris
            tracks[track] += 1

        
        test_playlists_tracks.append(tracks)
        
    
    print ("Data loaded. Creating features matrix")
    train_playlist_count =  train_playlists_count
    test_playlist_count = len(test_playlists)
    total_playlist_count =  train_playlist_count + test_playlist_count
    print("train:{}, test:{}, total:{}".format(train_playlist_count, test_playlist_count, total_playlist_count))

     
    dv = DictVectorizer()
    interaction_matrix = dv.fit_transform(playlists_tracks+[{}]*test_playlist_count) 
    print("train interaction matrix:{}".format(interaction_matrix.shape))
    

    model = LightFM(loss='warp', no_components=200, max_sampled=30, item_alpha=1e-06, user_alpha=1e-06, random_state=10)
    model = model.fit(interaction_matrix, epochs=150, num_threads=32)

    # freeze the gradient and optimize held-out users
    model.item_embedding_gradients = np.finfo(np.float32).max * np.ones_like(model.item_embedding_gradients)
    model.item_bias_gradients = np.finfo(np.float32).max * np.ones_like(model.item_bias_gradients)
    model.item_alpha = 0.0
    model.user_alpha = 0.0
    model.user_embedding_gradients[:train_playlist_count,:] = np.finfo(np.float32).max * np.ones_like(model.user_embedding_gradients[:train_playlist_count,:])
    model.user_bias_gradients[:train_playlist_count] = np.finfo(np.float32).max * np.ones_like(model.user_bias_gradients[:train_playlist_count])

    # Use the trained model to get a representation of the playlists on challenge set
    interaction_matrix = dv.transform(playlists_tracks+test_playlists_tracks)
    print("test interaction matrix:", interaction_matrix.shape)
    
    # print("test feature shape:{}, identity padding.shape:{}".format(playlist_features.shape, eye.shape)) 
    model.user_embeddings[-test_playlist_count:] = ((model.random_state.rand(test_playlist_count, model.no_components) - 0.5) / model.no_components).astype(np.float32)
    model = model.fit_partial(interaction_matrix, epochs=150, num_threads=32)
    print ("Model Trained")

    user_biases, user_embeddings = model.get_user_representations()
    item_biases, item_embeddings = model.get_item_representations()
    
    pickle.dump(item_embeddings, open(f'{dataset_path}CF_track_embeddings.pkl', "wb"))
    print("saved track embeddings to: {}".format(f'{dataset_path}CF_track_embeddings.pkl'))

    print("Generating Recommendations ")
    recommended_tracks = [] 
    for i, playlist in tqdm.tqdm(enumerate(test_playlists)):
        playlist_pos = train_playlists_count+i
        y_pred = user_embeddings[playlist_pos].dot(item_embeddings.T) + item_biases
        topn = np.argsort(-y_pred)[:len(test_playlists_tracks[i])+4000]
        rets = [(dv.feature_names_[t], float(y_pred[t])) for t in topn]
        recommended_tracks.append([x[0] for x in rets[:500]])

    rec_df = pd.DataFrame({'pid': test_playlists, 'recs': recommended_tracks})
    pickle.dump(rec_df, open(output_file, "wb"))

if __name__ == '__main__':
    b = time.time()
    exp_path = 'FILL IN '
    output_file = f'{exp_path}/small_100_10_CF_recs.pkl'
    process_mpd(output_file, 10)
    a = time.time()
    print("Process took {} min".format((a-b)/60))
