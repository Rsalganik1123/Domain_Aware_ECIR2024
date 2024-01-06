from collections import OrderedDict
from collections import namedtuple
import numpy as np
from scipy import stats
import pandas as pd 
import csv
import pickle 

#R precision
def r_precision(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions] 
    # Calculate metric
    target_set = set(targets)
    target_count = len(target_set)
    return float(len(set(predictions[:target_count]).intersection(target_set))) / target_count

# ​DCG
def dcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
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
    retrieved_elements = __get_unique(retrieved_elements[:k])
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))

def ndcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    
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
    idcg = dcg(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg
    
def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """
    return list(OrderedDict.fromkeys(original_list))

Metrics = namedtuple('Metrics', ['r_precision', 'ndcg', 'plex_clicks'])

# playlist extender clicks
def playlist_extender_clicks(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    i = set(predictions).intersection(set(targets))
    for index, t in enumerate(predictions):
        for track in i:
            if t == track:
                return float(int(index / 10))
    return float(max_n_predictions / 10.0 + 1)


def get_all_metrics(targets, predictions, k):
    return Metrics(r_precision(targets, predictions, k),
                ndcg(targets, predictions, k),
                playlist_extender_clicks(targets, predictions, k))

MetricsSummary = namedtuple('MetricsSummary', ['mean_r_precision',
                                               'mean_ndcg',
                                               'mean_plex_clicks',
                                               'coverage'])
                                               
                                               
def aggregate_metrics(ground_truth, sub, k, candidates):
    r_precision = []
    ndcg = []
    plex_clicks = []
    miss = 0
    cnt = 0
    for p in candidates:
        cnt += 1
        if p not in sub.keys() and str(p) not in sub.keys():
            miss += 1
            m = Metrics(0, 0, 0)  # TODO: make sure this is right
        else:
            m = get_all_metrics(ground_truth[p], sub[p], k) #sub[str(p)]
        r_precision.append(m.r_precision)
        ndcg.append(m.ndcg)
        plex_clicks.append(m.plex_clicks)

    cov = 1 - miss / float(cnt)
    print("MISSED:", miss, "OUT OF:", cnt)
    return MetricsSummary(
        stats.describe(r_precision).mean,
        stats.describe(ndcg).mean,
        stats.describe(plex_clicks).mean,
        cov
    )

def get_recs(): 
    path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/output_main_final_sum_window_10.csv'
    with open(path, "r") as f: 
        reader = csv.reader(f)
        counter = 0
        recs = {}
        for row in reader: 
            if counter == 0:
                counter += 1 
                continue 
            recs[row[0].strip()] = [int(r) for r in row[1:]]
            if counter == 1: break 
    return recs
          
def get_pkl_recs(path): 
    # data = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/output_main_final_sum_window_10.pkl', "rb"))
    # recs = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/output_main_final_sum_window_10_CF_small.pkl', "rb"))
    # recs = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/oct-6-2022/music_artist_genre_track_album_meta_FOCAL_LOSS/recs_TS1_top10000/CF_cosine_10_recommended_tracks.pkl', "rb"))
    # recs = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/oct-6-2022/meta_encodings_FOCAL_LOSS/recs_TS1_top10000/CF_cosine_10_recommended_tracks.pkl', "rb"))
    # recs = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/oct-6-2022/meta_encodings2_FOCAL_LOSS/recs_TS1_top10000/CF_cosine_10_recommended_tracks.pkl', "rb"))
    recs = pickle.load(open(path, "rb"))
    return dict(zip(recs.pid, recs.recs))
    # d = dict(zip(data.pid, data.recs))
    # print(d)

def get_gt(path): 
    # gt = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/cf_TS1.pkl', "rb"))
    # print(gt)
    # gt = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/small_test_cf.pkl', "rb"))
    gt = pickle.load(open(path, "rb"))
    return dict(zip(gt.pid, gt.track_uri)) #track_uris
      
def metric_launch(rec_path):
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/ablations_small_100_10/_no_isolate_random_LRDecay_hidden200_FOCAL_LOSS/CF_cosine_10_recommended_tracks.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/small_100_10_CF.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/GS/small_100_10_run/recs/cosine_recs_5.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/Majority/small_100_10/CF_recs.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/raw_feat/small_100_10/album+track/CF_recs.pkl'
    test_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/small_100_10/CF_test.pkl'
    gt_target = get_gt(test_path)
    sub = get_pkl_recs(rec_path) 
    pids = gt_target.keys() 
    # print(gt_target[0]) 
    # print(sub['0'])
    ms = aggregate_metrics(gt_target, sub, 500, pids)
    print(100, "500", ms)

def convert_format(data_path, rec_path, output_path): 
    data = pickle.load(open(data_path, "rb"))
    interaction_df, tracks_df, playlist_df = data['df_playlist'], data['df_track'], data['df_playlist_info']
    t = tracks_df[['tid', 'track_uri']]
    recs = pickle.load(open(rec_path, "rb"))
    tid_dict = dict(zip(tracks_df.tid, tracks_df.track_uri))
    recs['track_uri'] = recs.apply(lambda x: [tid_dict[tid] for tid in x['recs']], axis=1)
    recs.rename(columns={'recs': 'rec_tids', 'track_uri': 'recs', }, inplace=True)
    pickle.dump(recs, open(output_path, "wb"))

def convert_launch(rec_path, output_path): 
    data_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/small_100_10/train_val2.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/ablations_small_100_10/_no_isolate_random_LRDecay_hidden200_FOCAL_LOSS/recs_TS1_top1000/cosine_10_recommended_tracks.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/GS/small_100_10_run/recs/cosine_recs_5.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/Majority/small_100_10/recs.pkl'
    # output_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/GS/small_100_10_run/recs/CF_cosine_recs_5.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/GS/small_100_10_run/album_track_2560_10+25/recs/cosine_recs_9.pkl'
    # output_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/GS/small_100_10_run/album_track_2560_10+25/recs/CF_recs_9.pkl'
    # rec_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/raw_feat/small_100_10/album+track/recs.pkl'
    # output_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/benchmarks/raw_feat/small_100_10/album+track/CF_recs.pkl'
    
    convert_format(data_path, rec_path, output_path)

exp_path = 'FILL IN'
rec_path = f'{exp_path}cosine_10_recommended_tracks.pkl'
output_path = f'{exp_path}/CF_cosine_10_recommended_tracks.pkl'
convert_launch(rec_path, output_path)
metric_launch(output_path) 
