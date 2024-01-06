import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import torch 
import pickle 

def artist_diversity(rec_df, k):
    '''
    INPUT: 
        rec_df: dataframe with row: (pid, recs, tid, arid, appear_pop)
        k: number of recommended tracks per pid 
    METHOD: 
        aggregates the number of arids associated with each playlist, returns average of raw count and average of normalized count 
    OUTPUT: 
        average of unique arids recommended per playlist, average of unique arids recommended per playlist normalized by number of recommendations per playlist
    '''
    div = rec_df.groupby('pid')['arid'].apply(lambda x: len(set(x))).reset_index(name='#arid')
    div['av_#arid'] = div['#arid'].apply(lambda x: x/k) #.reset_index(name='av_#arid')
    #return np.mean(div['#arid']), np.mean(div['av_#arid'])
    return div['#arid'], div['av_#arid']

def item_LT_metrics(rec_df,  k, LT_col, LT=0):
    '''
    INPUT: 
        rec_df: dataframe with row: (pid, recs, tid, arid, appear_pop)
        k: number of recommended tracks per pid
        LT: threshold bin considered for an item to be considered as long tail 
    METHOD: 
        aggregates the number of LT_items associated with each playlist, returns average of raw count and average of normalized count 
    OUTPUT: 
        average of LT_items recommended per playlist (cvg LT), average of LT_items recommended per playlist normalized by number of recommendations per playlist (% LT)
    '''
    LT_df = rec_df.groupby('pid')[LT_col].apply(lambda x: len([i for i in x if i <=LT])).reset_index(name='#LT')
    LT_df['%LT'] = LT_df['#LT'].apply(lambda x: x/k) #.reset_index(name='#LT')
    pop = rec_df.groupby('pid')[LT_col].apply(lambda x: np.mean(x)).reset_index(name='av_pop')
    LT_df = pd.merge(LT_df, pop, on='pid', how='left')
    # return  np.mean(LT_df['av_pop']), np.mean(LT_df['%LT']), np.mean(LT_df['#LT'])
    return LT_df['av_pop'], LT_df['%LT'], LT_df['#LT']

def item_cvg(rec_df, df_items, LT_col, LT=0): 
    df_items = df_items.astype({LT_col: 'int32'})
    LT_in_recs = rec_df[rec_df[LT_col] <= LT].tid.unique()
    LT_in_dataset = df_items[df_items[LT_col] <= LT].tid.unique()
    LT_cvg = len(LT_in_recs)/len(LT_in_dataset) 
    
    tid_in_recs = rec_df.tid.unique()
    tid_in_dataset = df_items.tid.unique()
    tid_cvg = len(tid_in_recs)/len(tid_in_dataset) 

    return LT_cvg, tid_cvg 
     
def artist_cvg(rec_df, df_items): 
    arid_in_recs = rec_df.arid.unique()
    arid_in_dataset = df_items.arid.unique()
    arid_cvg = len(arid_in_recs)/len(arid_in_dataset)
    return arid_cvg 

def sound_homogeneity(rec_df, df_items): 
    
    mus_feat  = ['danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo']
    recs_with_mus = pd.merge(df_items[['tid'] + mus_feat], rec_df, on="tid", how='right')
    grouped_recs_with_mus = recs_with_mus.groupby('pid').apply(lambda x: x[mus_feat].values.astype(int)).reset_index(name='mus_feat')
    grouped_recs_with_mus['homo'] = grouped_recs_with_mus['mus_feat'].apply(lambda x: cosine_sim(x))
    
    return grouped_recs_with_mus['homo']

def apply_k(rec_df, k): 
    rec_df['recs'] = rec_df['recs'].map(lambda x: x[:k])
    return rec_df

def cosine_sim(x):
    x = torch.tensor(x).float()
    a_norm = x / x.norm(dim=1)[:, None]
    res = torch.mm(a_norm, a_norm.transpose(0,1))
    return torch.mean(res)

def prep_recs(recs, df_items, k, LT_col): 
    recs = apply_k(recs, k)
    expanded_recs = recs.explode('recs', ignore_index=True)
    expanded_recs['tid'] = expanded_recs.recs
    recs_arid = pd.merge(expanded_recs, df_items[['tid', 'arid', LT_col]], on='tid', how='left')
    recs_arid = recs_arid.astype({LT_col: 'int32'})
    return recs_arid 

def get_metrics(rec_df, df_items, k, LT_col, LT_bin, verbose = True, output_path=None): 
    recs = prep_recs(rec_df, df_items, k, LT_col)
    raw_diversity, norm_diversity = artist_diversity(recs, k)
    playlist_homogeneity = sound_homogeneity(recs, df_items)
    av_pop, perc_LT, count_LT = item_LT_metrics(recs, k, LT_col, LT=LT_bin)
    LT_cvg, tid_cvg = item_cvg(recs, df_items, LT_col, LT=LT_bin)
    arid_cvg = artist_cvg(recs, df_items)

    if verbose: 
        breakdown_df = pd.DataFrame({
                        'pid': recs.pid.unique(), 
                        'norm_diversity': norm_diversity.values,
                        'sound_homogeneity': playlist_homogeneity.values,  
                        'perc_LT': perc_LT.values 
                        })
        pickle.dump(breakdown_df, open(output_path + f'{LT_col}_fairness_breakdown_by_pid.pkl', "wb")) 
        print("***saving verbose breakdown to:{}***".format(output_path + f'{LT_col}_fairness_breakdown_by_pid.pkl'))
    
    return {
        'raw_diversity': np.mean(raw_diversity), 
        'norm_diversity': np.mean(norm_diversity), 
        'sound_homogeneity': np.mean(playlist_homogeneity), 
        'av_pop': np.mean(av_pop), 
        'perc_LT': np.mean(perc_LT), 
        'count_LT': np.mean(count_LT), 
        'tid_cvg': tid_cvg, 
        'arid_cvg': arid_cvg, 
        'LT_item_cvg': LT_cvg
    }

def fairness_eval(cfg, k, rec_df, output_path = None, LT_bin=0, mode='PS', setting = 'test'):
    # print("Mode for fairness metrics is: {}".format(mode)) 
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    if setting == 'test': 
        test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, "rb"))
    else: 
        valid_idx = all_data['val_indices']
        val_data = df_interactions.loc[valid_idx]
        val_pid = val_data.groupby('pid')['tid'].apply(list).reset_index(name='tids')
        val_pid['num'] = val_pid['tids'].apply(len)
        val_100 = val_pid[val_pid.num >= 100]
        val_100['tids'] = val_100['tids'].apply(lambda x: x[:100])
        val_100['num'] = val_100['tids'].apply(len)
        val_100 = val_100.explode('tids').drop(columns=['num'])
        test_set = val_100.rename(columns={'tids':'tid'})
    test_set = pd.DataFrame({'pid': test_set.pid.unique(), 'recs':test_set.groupby('pid')['tid'].apply(list).tolist()}) 
    
    LT_col = cfg.FAIR.POP_FEAT #if cfg.FAIR.POP_FEAT != None else 'appear_pop'
    print("LT col:{}, LT_bin:{}".format(LT_col, LT_bin))
    pinsage_rec_fairness = get_metrics(rec_df, df_items, k, LT_col, LT_bin, verbose=True, output_path=output_path)
    
    if mode == 'TEST': 
        test_rec_fairness = get_metrics(test_set, df_items, k, LT_col)
        print(test_rec_fairness)
    return pinsage_rec_fairness, None, None
