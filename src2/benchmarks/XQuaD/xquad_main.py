import pickle 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import time
# import ipdb



def get_user_pref(gt_with_pop_info, LT_col, gen_amount=None): 
    data = []  
    # ipdb.set_trace() 
    for pid in gt_with_pop_info.pid.unique(): 
        tracks = gt_with_pop_info[gt_with_pop_info.pid == pid]
        if gen_amount: 
            tracks = gt_with_pop_info[gt_with_pop_info.pid == pid][:gen_amount]
        LT = len(tracks[tracks[LT_col] == 0])/gen_amount 
        nLT = len(tracks[tracks[LT_col] > 0])/gen_amount 
        data.append([pid, LT, nLT]) 
    user_pref_df = pd.DataFrame(data = data, columns = ['pid', 'LT', 'nLT'])
    return user_pref_df


def get_list_state(S, pop_info, LT_col, grp): 
    if len(S) == 0:
        LT_ratio = 0 #Check this  
    else: 
        LT_ratio = len(pop_info[(pop_info.tid.isin(S)) & (pop_info[LT_col] == grp)])/len(S)
    return LT_ratio

 
def test_metrics(k,recs, LT_col, LT = 0):

    #calculate % LT in XQUAD
    LT_in_x = recs.groupby('pid')[LT_col].apply(lambda x: len([i for i in x if i <=LT])).reset_index(name='#LT')
    LT_in_x['%LT'] = LT_in_x['#LT'].apply(lambda x: x/k)

    return np.mean(LT_in_x['%LT'])

def test_xquad(gen_amount = 5, length = 3, lmbda = 0.1): #gen_amount = 5, length = 3, lmbda = 1.0
    # ipdb.set_trace() 
    #EXAMPLE
    np.random.seed(5)


    track_data = pd.DataFrame({
        'tid': list(range(15)),
        'log10_popcat': np.random.choice([0,1], size=15)})
    gt = pd.DataFrame({
        'pid': [5,6,7], 
        'tid': [np.random.choice(15, 5, replace = False) for i in range(3)] })
    recs_input = pd.DataFrame({
        'pid': [5,6,7], 
        'recs': [np.random.choice(15, 5, replace = False) for i in range(3)], 
        'scores': [np.random.rand(5) for i in range(3)] })
    

    output_path = None 

    LT_col = 'log10_popcat'
    pop_info = track_data[['tid', LT_col]]
    recs = recs_input.explode(['recs', 'scores']).rename(columns={'recs':'tid'})
    recs_with_pop_info = pd.merge(recs, pop_info).astype({'pid': 'int', 'tid': "int", 'scores':'float'}).sort_values('pid')
    gt = gt.explode('tid')
    gt_with_pop_info = pd.merge(gt, pop_info)
    
    # ipdb.set_trace() 
    print("Calculating user preferences")
    #Calculate user preferences 
    user_prefs = get_user_pref(gt_with_pop_info, LT_col, gen_amount = gen_amount)
    tid_lists, score_lists = [] , [] 
    for pid in tqdm(recs_with_pop_info.pid.unique()): 
        tids, scores = [] , []
        u_pref = user_prefs[user_prefs.pid == pid ]
        rec_df = recs_with_pop_info[recs_with_pop_info.pid == pid].copy() 
        for i in range(length):
            #generate options 
            options = rec_df[~rec_df.tid.isin(tids)].copy()
            #get scores  
            rel = options['scores'].values 
            LT = options[LT_col].values
            LT_ratio, nLT_ratio = (1-get_list_state(tids, pop_info, LT_col, 0)), (1-get_list_state(tids, pop_info, LT_col, 1))
            # ipdb.set_trace()
            xquad_scores = [(1-lmbda)*rel[j] + lmbda*(u_pref['LT'].values * float(LT[j] == 0) * LT_ratio + u_pref['nLT'].values *float(LT[j] == 1)* nLT_ratio) for j in range(len(rel))] 
            max_idx = np.argmax(xquad_scores)
            tid = options['tid'].values[max_idx]
            tids.append(int(tid))
            scores.append(xquad_scores[max_idx][0]) 
        tid_lists.append(tids)
        score_lists.append(scores)
    xquad_recs = pd.DataFrame({'pid': recs_with_pop_info.pid.unique(), 'recs': tid_lists, 'scores': score_lists})
    # ipdb.set_trace()
    print(xquad_recs)
    xquad_recs_exploded = xquad_recs.explode(['recs', 'scores']).rename(columns ={'recs':'tid'})
    xquad_recs_with_pop = pd.merge(xquad_recs_exploded, pop_info, on='tid').rename({'tid':'recs'})
    LT_x = test_metrics(k = length, recs = xquad_recs_with_pop, LT_col = LT_col)

    #Test OG list 
    recs_input['recs'] = recs_input['recs'].map(lambda x: list(x[:length]))
    expanded_recs = recs_input.explode('recs', ignore_index=True)
    expanded_recs['tid'] = expanded_recs.recs
    recs_input_with_LT = pd.merge(expanded_recs, pop_info)
    LT_og = test_metrics(k = length, recs = recs_input_with_LT, LT_col = LT_col)
    
    print("LONG TAIL METRICS: OG:{}, X:{}".format(LT_og, LT_x)) 

    if output_path: 
        print("writing xquad recommendation list to:{}".format(output_path))
        pickle.dump(xquad_recs, open(output_path, "wb"))


def launch_xqad_run(p_rec, p_gt, p_data, output_path,lmbda, LT_col, length=100, gen_amount = 10): 
    b = time.time()
    #Load data 
    print("Loading data")
    train_data = pickle.load(open(p_data, "rb"))
    user_data = train_data['df_playlist_info']
    track_data = train_data['df_track']
    gt = pickle.load(open(p_gt, "rb"))
    recs = pickle.load(open(p_rec, "rb"))

    # ipdb.set_trace() 
    #EXAMPLE
    # np.random.seed(5)
    # track_data = pd.DataFrame({
    #     'tid': list(range(15)),
    #     'LT_grp': np.random.choice([0,1], size=15)})

    # gt = pd.DataFrame({
    #     'pid': [5,6,7], 
    #     'tid': [np.random.choice(15, 5, replace = False) for i in range(3)] })
    # recs = pd.DataFrame({
    #     'pid': [5,6,7], 
    #     'recs': [np.random.choice(15, 5, replace = False) for i in range(3)], 
    #     'scores': [np.random.rand(5) for i in range(3)] })
    
    
    # LT_col = 'LT_grp' 
    # LT_col = 'log10_popcat'
    pop_info = track_data[['tid', LT_col]]
    recs = recs.explode(['recs', 'scores']).rename(columns={'recs':'tid'})
    gt = gt.explode('tid')
    recs_with_pop_info = pd.merge(recs, pop_info).astype({'pid': 'int', 'tid': "int", 'scores':'float'}).sort_values('pid')
    gt_with_pop_info = pd.merge(gt, pop_info)
 

    # print(recs_with_pop_info)
    print("Calculating user preferences")
    #Calculate user preferences 
    user_prefs = get_user_pref(gt_with_pop_info, LT_col, gen_amount = gen_amount)
    tid_lists, score_lists = [] , [] 
    for pid in tqdm(recs_with_pop_info.pid.unique()): 
        tids, scores = [] , []
        u_pref = user_prefs[user_prefs.pid == pid ]
        rec_df = recs_with_pop_info[recs_with_pop_info.pid == pid].copy() 
        for i in range(length):
            #generate options 
            options = rec_df[~rec_df.tid.isin(tids)].copy()
            #get scores  
            rel = options['scores'].values 
            LT = options[LT_col].values
            LT_ratio, nLT_ratio = (1-get_list_state(tids, pop_info, LT_col, 0)), (1-get_list_state(tids, pop_info, LT_col, 1))
            # xquad_scores = [(1-lmbda)*rel[j] + lmbda*(u_pref['LT'].values * (1-LT[j])* LT_ratio + u_pref['nLT'].values * (1-LT[j])* nLT_ratio) for j in range(len(rel))] 
            xquad_scores = [(1-lmbda)*rel[j] + lmbda*(u_pref['LT'].values * float(LT[j] == 0) * LT_ratio + u_pref['nLT'].values *float(LT[j] == 1)* nLT_ratio) for j in range(len(rel))] 
            max_idx = np.argmax(xquad_scores)
            tid = options['tid'].values[max_idx]
            tids.append(int(tid))
            scores.append(xquad_scores[max_idx][0]) 
        tid_lists.append(tids)
        score_lists.append(scores)
    xquad_recs = pd.DataFrame({'pid': recs_with_pop_info.pid.unique(), 'recs': tid_lists, 'scores': score_lists})
    print(xquad_recs)
    print("writing xquad recommendation list to:{}".format(output_path))
    if output_path: 
        pickle.dump(xquad_recs, open(output_path, "wb"))

    a = time.time() 
    print("Process took {} min".format((a-b)/60))

