
from src2.eval.gen_embeddings import *
from src2.eval.gen_recommendations import *
from src2.eval.feature_benchmarks import *  
from src2.eval.competition_metrics import *  
from src2.fairness.fairness_metrics import * 
from src2.utils.save_res import * 
# from src2.utils.misc import * 
# from src2.model.layers import UsertoItemScorer
from tqdm import tqdm 
from scipy import stats 
from glob import glob 

def validation_macr(cfg, val_gt, val_recs, output_path, k=100): 
    
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, "rb"))
    df_items = all_data[cfg.DATASET.ITEM_DF]
    LT_col = cfg.FAIR.POP_FEAT
    gen_amount = 10 
    r_prec, competition_ndcg = [] , [] 
    prec_at_500, prec_at_1K, prec_at_10K, artist_prec= [], [] , [], [] 
    recall_at_500, recall_at_1K, recall_at_10K, artist_prec= [], [] , [], []   
    normal_ndcg_at_500, normal_ndcg_at_1K, normal_ndcg_at_10K = [], [], [] 
    ndcg_all = [] 
    test_set = val_gt.rename(columns={'uid': 'pid', 'sid': 'tid'})
    recommended_tracks = val_recs 

    for p in tqdm(test_set.pid.unique()):
        associated_tracks = test_set[test_set.pid == p]['tid']
        recs = recommended_tracks[recommended_tracks.pid == p]['recs'].tolist()[0] 
        r_prec.append(r_precision(recs, gen_amount, associated_tracks))
        competition_ndcg.append(compute_ndcg(recs, associated_tracks))
        artist_prec.append(r_precision_artist(recs, gen_amount, associated_tracks, df_items))

    recs = prep_recs(val_recs, df_items, k, LT_col)
    raw_diversity, norm_diversity = artist_diversity(recs, k)
    playlist_homogeneity = sound_homogeneity(recs, df_items)
    av_pop, perc_LT, count_LT = item_LT_metrics(recs, k, LT_col, LT=0)
    LT_cvg, tid_cvg = item_cvg(recs, df_items, LT_col, LT=0)
    arid_cvg = artist_cvg(recs, df_items)

    performance_metrics = {
        'r_precision': np.mean(r_prec),  
        'competition_ndcg': np.mean(competition_ndcg), 
        'artist_prec': np.mean(artist_prec)} 
    fairness_metrics = {
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

    if output_path: 
        save_results(output_path,'', 'perf', performance_metrics)
    if output_path: 
        save_results(output_path, LT_col, 'fair', fairness_metrics)
  
def launch_performance_eval_clean(cfg, gen_amount, epoch, recommended_track_path, mode = 'test', output_path=None, verbose=True): 
    print("*** Running Competition Metric Eval***")
    recommended_tracks = pickle.load(open(recommended_track_path, 'rb'))
    competition_performance = competition_eval(cfg, recommended_tracks, gen_amount, output_path, verbose=verbose, mode=mode)
    competition_performance['gen_amount'] = gen_amount 
    competition_performance['epoch'] = epoch 
    for v in competition_performance.keys(): 
        print(v, competition_performance[v])
    if output_path: 
        save_results(output_path,'', 'perf', competition_performance )

def launch_fairness_audit_clean(cfg, k, epoch, recommended_track_path, setting = 'test', mode ='PS', LT_bin = 0, output_path=None):
    print("*** Running Fairness Audit***")
    recommended_tracks = pickle.load(open(recommended_track_path, "rb"))
    this_fairness_performance, deviation_fairness_performance, test_fairness_performance = fairness_eval(cfg, k, recommended_tracks, mode =mode, setting = setting, LT_bin=LT_bin, output_path = output_path)
    this_fairness_performance['k'] = k
    this_fairness_performance['LT_bin'] = LT_bin
    this_fairness_performance['epoch'] = epoch
    for v in this_fairness_performance.keys():
        print(v, this_fairness_performance[v])
    if output_path: 
        save_results(output_path, cfg.FAIR.POP_FEAT, 'fair', this_fairness_performance)
    
def launch_performance_eval(cfg, gen_amount, epoch, recommended_tracks = None, recommended_track_path = None, output_path=None, verbose=False): 
    print("*** Running Competition Metric Eval***")
    if recommended_track_path: 
        recommended_tracks = pickle.load(open(recommended_track_path, 'rb'))
    competition_performance = competition_eval(cfg, recommended_tracks, gen_amount, output_path, verbose=verbose)
    competition_performance['gen_amount'] = gen_amount 
    competition_performance['epoch'] = epoch 
    for v in competition_performance.keys(): 
        print(v, competition_performance[v])
    if output_path: 
        save_results(output_path,recommended_track_path.split('/')[-1].split('.pkl')[0], 'perf', competition_performance )

def launch_fairness_audit(cfg, k, epoch, reccommended_tracks = None, recommended_track_path=None, exposure_threshold=5, mode ='PS', output_path=None):
    print("*** Running Fairness Audit***")
    if recommended_track_path: 
        recommended_tracks = pickle.load(open(recommended_track_path, "rb"))
    this_fairness_performance, deviation_fairness_performance, test_fairness_performance = fairness_eval(cfg, k, recommended_tracks, exposure_threshold, mode)
    this_fairness_performance['k'] = k
    this_fairness_performance['epoch'] = epoch
    print("THIS RUN PERFORMANCE")
    for v in this_fairness_performance.keys():
        print(v, this_fairness_performance[v])
    if deviation_fairness_performance: 
        print("DEVIATION PERFORMANCE")
        for v in deviation_fairness_performance.keys():
            print(v, deviation_fairness_performance[v])
        for v in test_fairness_performance.keys():
            print(v, test_fairness_performance[v])
    if output_path: 
        save_results(output_path,recommended_track_path.split('/')[-1].split('.pkl')[0], 'fair', this_fairness_performance)
    