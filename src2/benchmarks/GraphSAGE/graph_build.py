import pickle 
import dgl 
import torch 
import numpy as np 
from src2.utils.build_utils.dgl_builder import PandasGraphBuilder


dataset_path = 'FILL IN'

def get_feat(): 
    return 0 


def build(): 
    print("loading data")
    path = f'{dataset_path}datasets/small_100_10/train_val.pkl'
    data = pickle.load(open(path, "rb")) 
    print('data loaded')
    df_items = data['df_track']
    df_users = data['df_playlist_info']
    df_interactions = data['df_playlist']
    train_indices = data['train_indices']
    val_indices = data['val_indices']

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, 'tid', 'track')
    graph_builder.add_entities(df_users, 'pid', 'playlist')
    graph_builder.add_binary_relations(df_interactions, 'pid', 'tid', 'contains')
    graph_builder.add_binary_relations(df_interactions, 'tid', 'pid', 'contained_by')
    g = graph_builder.build()
    print(g)
    
    # feats = ['tid', 'alid', 'arid', 'danceability', 'energy', 'loudness',
    #     'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    #     'valence', 'tempo']
    feats = ['img_emb', 'track_name_emb']
    track_feats = torch.tensor(np.concatenate([np.asarray(list(df_items[k].values)) for k in feats], axis=1)) #vectors
    # track_feats = torch.tensor(np.stack(df_items[feats].values)).float() #values 
    
    playlist_feats = torch.zeros((g.num_nodes('playlist'), track_feats.shape[1])) 
    g.ndata['feat']={
        'playlist': playlist_feats, 
        'track': track_feats
        }
    
    
    homo_g = dgl.to_homogeneous(g, ndata=['feat'], store_type=True, return_count=False)
    print(homo_g)
    i = homo_g.edata['_TYPE'].nonzero().flatten()
    reverse_eids = homo_g.edges(form='eid')[i]
    return homo_g, reverse_eids, torch.tensor(train_indices), torch.tensor(val_indices) 
