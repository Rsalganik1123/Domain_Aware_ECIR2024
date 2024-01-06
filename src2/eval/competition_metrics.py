import numpy as np
import pickle 
from tqdm import tqdm 
import pandas as pd 
from collections import OrderedDict
# import ipdb 


'''
File contains a series of metrics used in the RecSys 2018 competition. 
At the baseline these are RPrecision, NDCG, and ClickCount. 
Then, they are broken down into all, and strict. 
Strict - used for @k experiments as in, it will limit the number of recommendations which can be considered to evaluate
All - Includes even the values used for generating a playlist recommendation 

Each function takes three inputs: 
    preds: like of tids for predicted tracks 
    known_num: number of tracks used to generate a playlist embedding 
    all_tids: all the tids associated with a playlist (ground truth)
'''

# log_bottom = np.log2(np.arange(2, 501))


def compute_clicks(preds, known_num, all_tids):
    tids_known = set(all_tids[:known_num])
    tids_rest = set(all_tids[known_num:])
    preds = [x for x in preds if x not in tids_known]
    preds = [(idx, x) for idx, x in enumerate(preds)]
    overlap = [x for x in preds if x[1] in tids_rest]
    if len(overlap) > 0:
        return max(0, (overlap[0][0] - 1) / 10)
    else:
        return 50


def compute_clicks_all(preds, known_num, all_tids):
    preds = [(idx, x) for idx, x in enumerate(preds)]
    overlap = [x for x in preds if x[1] in all_tids]
    if len(overlap) > 0:
        return max(0, (overlap[0][0] - 1) / 10)
    else:
        return 50

def dcg(relevant_elements, retrieved_elements):
    """Compute the Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Returns:
        DCG value
    """
    retrieved_elements = __get_unique(retrieved_elements)
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))

def compute_ndcg(retrieved_elements,  relevant_elements):
    
    """Compute the Normalized Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.
    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Returns:
        NDCG value
    """
    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    # ipdb.set_trace() 
    idcg = dcg(
        relevant_elements, relevant_elements)
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements)
    return true_dcg / idcg
    
def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """
    return list(OrderedDict.fromkeys(original_list))


def normal_ndcgk(pred_tids, gt_tids, k):
    preds = pred_tids[:k]
    relevances = np.array([x in gt_tids for x in preds])
    log_bottom = np.log2(np.arange(3, len(relevances)+2))
    dcg = relevances[0] + np.sum(relevances[1:] / log_bottom)
    idcg = 1 + np.sum(np.ones(len(gt_tids)-1) / log_bottom[:len(gt_tids)-1])
    ndcg = dcg / idcg
    return ndcg

def recallk(pred_tids, known_num, gt_tids, k):
    gt = gt_tids
    preds = pred_tids[:k]
    overlap = [i for i in gt if i in preds]
    # overlap = np.intersect1d(gt, pred_tids[:k])
    portion = len(overlap) / len(gt)
    return portion

def precisionk(pred_tids, known_num, gt_tids, k):
    gt = gt_tids
    preds = pred_tids[:k]
    overlap = [i for i in gt if i in preds]
    portion = len(overlap) / k
    return portion

def r_precision(pred_tids, known_num, gt_tids):
    gt = gt_tids
    preds = pred_tids[:len(gt)]
    overlap = [i for i in gt if i in preds]
    # overlap = np.intersect1d(gt, pred_tids[:len(gt)])
    portion = len(overlap) / len(gt)
    return portion

def r_precision_artist(pred_tids, known_num, gt_tids, tracks): 
    gt = gt_tids
    preds = pred_tids[:len(gt)]
    gt_artists = tracks[tracks.tid.isin(gt)].arid.unique() 
    rec_artists =  tracks[tracks.tid.isin(preds)].arid.unique()
    overlap = np.intersect1d(gt_artists, rec_artists)
    portion = len(overlap)  / len(gt_artists)
    return portion 

def r_precision_album(pred_tids, known_num, gt_tids, tracks): 
    entries = gt_tids[:known_num]
    gt_albums = tracks[tracks.tid.isin(entries)].alid.unique() 
    rec_albums =  tracks[tracks.tid.isin(pred_tids)].alid.unique()
    overlap = np.intersect1d(gt_albums, rec_albums)
    portion = len(overlap) / len(gt_albums)
    return portion 


def competition_eval(cfg, recommended_tracks, gen_amount, output_path, verbose=False, mode = 'test'): # (preds, known_num, all_tids, tracks): 
    print("verbose mode:{}".format(verbose))
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, "rb"))
    df_items = all_data[cfg.DATASET.ITEM_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    if mode == 'test': 
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
    r_prec, competition_ndcg = [] , [] 
    prec_at_500, prec_at_1K, prec_at_10K, artist_prec= [], [] , [], [] 
    recall_at_500, recall_at_1K, recall_at_10K, artist_prec= [], [] , [], []   
    normal_ndcg_at_500, normal_ndcg_at_1K, normal_ndcg_at_10K = [], [], [] 
    ndcg_all = []  
    for p in tqdm(test_set.pid.unique()):
        associated_tracks = test_set[test_set.pid == p]['tid']
        recs = recommended_tracks[recommended_tracks.pid == p]['recs'].tolist()[0] 
        r_prec.append(r_precision(recs, gen_amount, associated_tracks))
        competition_ndcg.append(compute_ndcg(recs, associated_tracks))
        artist_prec.append(r_precision_artist(recs, gen_amount, associated_tracks, df_items))
        
        
    if verbose and output_path: 
        
        metrics_df = pd.DataFrame({'pid':test_set.pid.unique(), 'r_precision': r_prec,  'competition_ndcg': competition_ndcg, 'artist_prec': artist_prec 
        
        })
        pickle.dump(metrics_df, open(output_path + 'performance_breakdown_by_pid.pkl', "wb"))
        print("***saving verbose breakdown to:{}***".format(output_path + 'performance_breakdown_by_pid.pkl'))
    
    performance_metrics = {
        'r_precision': np.mean(r_prec),  
        'competition_ndcg': np.mean(competition_ndcg), 
        'artist_prec': np.mean(artist_prec),  
    }
    return performance_metrics