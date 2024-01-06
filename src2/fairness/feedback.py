import pickle 
import pandas as pd 

def apply_k(rec_df, k): 
    rec_df['recs'] = rec_df['recs'].map(lambda x: x[:k])
    return rec_df

def augment_dataset(data_path, rec_path, round, output_path, aug_amt = 10): 
    data = pickle.load(open(data_path, "rb"))
    tracks = data['df_track']
    interactions = data['df_playlist']
    playlists = data['df_playlist_info']
    recs = pickle.load(open(rec_path, "rb"))
    recs = apply_k(recs, aug_amt)
    new_pids = list(range(len(recs.pid.unique()))) + max(playlists.pid.unique()) + 1 
    playlists_new = pd.DataFrame({'name': ['r{}_aug_p{}'.format(round, i) for i in range(len(new_pids))] , 'pid': new_pids})
    playlists_plus = pd.concat([playlists, playlists_new], ignore_index=True)
    old_pids = list(recs.pid.unique())
    pid_dict = dict(zip(old_pids, new_pids))
    interact_new = recs.explode('recs').copy() 
    interact_new['pid'] = recs.apply(lambda x: pid_dict[x['pid']], axis=1) 
    interact_new['pos'] = list(range(aug_amt)) * len(recs.pid.unique()) 
    interact_new = interact_new.rename(columns = {'recs': 'tid'})
    interact_new = interact_new.reset_index(drop=True)
    interactions_plus = pd.concat([interactions, interact_new], ignore_index=True)
    train_idx_new = interactions_plus[interactions_plus.pid.isin(new_pids)].index
    train_idx = data['train_indices']
    train_idx_plus = train_idx.append(train_idx_new)
    new_data = {
    'df_track': data['df_track'], 
    'df_playlist_info': playlists_plus,
    'df_playlist': interactions_plus, 
    'train_indices': train_idx_plus, 
    'val_indices': data['val_indices'], 
    'test_set': data['test_set']
    }
    print("UPDATED DATASET: before... plists:{}, tracks:{}, interact:{}, \nnow ... plists:{}, tracks:{}, interact:{}".format(
        len(playlists.pid.unique()), len(tracks.tid.unique()), len(interactions), 
        len(playlists_plus.pid.unique()), len(tracks.tid.unique()), len(interactions_plus)
    ))
    pickle.dump(new_data, open(output_path, "wb"))
    print("Saved to :{}".format(output_path))
    
