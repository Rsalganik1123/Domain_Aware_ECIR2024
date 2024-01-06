from typing import NamedTuple
import glob
import os  
from itertools import product
# import glob.glob 

scratch_dir = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/oct-6-2022/'
# scratch_dir2 = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/ablation_10000_100/Benchmark/'
# scratch_dir2 = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/ablations_small_100_10/fairness_hp/repro/'
scratch_dir2 = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/LFM/subset/NDCG_exp/'

temp_result_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/temp_results/'
test_set_dict = {
    'TS1': '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set1_clean.pkl', 
    'TS2': '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set2_clean.pkl'
}

test_set_paths = [
    '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set1_clean.pkl',
    '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set2_clean.pkl'
]

## Helpers for Launching PinSAGE Fair Train ### 

class ArgsforFairPSRuns(NamedTuple): 
    emb: list 
    projection_feat: list 
    projection_concat: list 
    hidden_size: int 
    fair_feat_set: int 
    output_path: str 
    gamma: int 
    alpha: int 
    boost: float 
    method: str 
    pop_feat: str 
     

music_embs = [['danceability', 16],
    ['energy', 16],
    ['loudness', 16],
    ['speechiness', 16],
    ['acousticness', 16],
    ['instrumentalness', 16],
    ['liveness', 16],
    ['valence', 16], 
    ['tempo', 16]]
mus_feat = ['tempo', 'liveness', 'instrumentalness', 'speechiness', 'loudness', 'acousticness', 'danceability', 'valence', 'energy']

# args_for_pinsage_FAIR_train = [ 
#     #GAMMA

    # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "PS_ScoreReg/v1_"), gamma=0.1, alpha=0.0, boost=0.0, method='vanilla', pop_feat=' '),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.1, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.1, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.1, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.1, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.2, alpha=0.0, boost=0.01),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.2, alpha=0.0, boost=0.01),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.2, alpha=0.0, boost=0.01),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.2, alpha=0.0, boost=0.01),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.2, alpha=0.0, boost=0.01),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.3, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.3, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.3, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.3, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.3, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.4, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.4, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.4, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.4, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.4, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.5, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.5, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.5, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.5, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.5, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.6, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.6, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.6, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.6, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.6, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.7, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.7, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.7, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.7, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.7, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.8, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.8, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.8, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.8, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.8, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=0.9, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=0.9, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=0.9, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=0.9, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=0.9, alpha=0.0, boost=0.0),
    
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v1_"), gamma=1.0, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v2_"), gamma=1.0, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v3_"), gamma=1.0, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v4_"), gamma=1.0, alpha=0.0, boost=0.0),
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "Benchmark/GAMMA_SPREAD/v5_"), gamma=1.0, alpha=0.0, boost=0.0),

    
#     # #Gamma 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v4_"), gamma=1.0, alpha=0.01, boost=0.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v5_"), gamma=1.0, alpha=0.01, boost=0.0), 
    
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v4_"), gamma=0.7, alpha=0.001, boost=0.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v5_"), gamma=0.7, alpha=0.001, boost=0.0), 

#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v4_"), gamma=0.2, alpha=0.001, boost=0.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "GAMMA/v5_"), gamma=0.2, alpha=0.001, boost=0.0), 
    
#     # #Alpha
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "ALPHA/v4_"), gamma=0.7, alpha=0.005, boost=0.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "ALPHA/v5_"), gamma=0.7, alpha=0.005, boost=0.0), 

#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "ALPHA/v4_"), gamma=0.7, alpha=0.01, boost=0.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "ALPHA/v5_"), gamma=0.7, alpha=0.01, boost=0.0), 

#     #OLD
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"), gamma=0.7, alpha=0.001, boost=0.01), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"), gamma=0.7, alpha=0.001, boost=0.0),
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"), gamma=0.7, alpha=0.001, boost=1.0), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"), gamma=0.7, alpha=0.001, boost=0.0)
    
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat, projection_concat=mus_feat, hidden_size=144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "allmus_allmus")), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img+track+mus_img")),
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "img+track+mus_track")), 
    
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['energy'], output_path=os.path.join(scratch_dir2, "img+track+mus_energy")), 
#     # ArgsforFairPSRuns(emb=[], projection_feat=['img_emb'], projection_concat=[], hidden_size=2048, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img_img")),
#     # ArgsforFairPSRuns(emb=[], projection_feat=['track_name_emb'], projection_concat=[], hidden_size=512, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "track_track")),
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=['img_emb'], projection_concat=[], hidden_size=144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "mus_dance")), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus")),
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "img+track+mus_dance")), 
#     # ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['energy'], output_path=os.path.join(scratch_dir2, "img+track+mus_energy")), 
# ]


music_embs2 = [
    ['danceability_10cat', 8],
    ['energy_10cat', 8],
    ['loudness_10cat', 8],
    ['speechiness_10cat', 8],
    ['acousticness_10cat', 8],
    ['instrumentalness_10cat', 8],
    ['liveness_10cat', 8],
    ['valence_10cat', 8], 
    ['tempo_10cat', 8],
]

all_feat_set = ['danceability_10cat', 'energy_10cat', 'loudness_10cat',
       'speechiness_10cat', 'acousticness_10cat', 'instrumentalness_10cat', 'liveness_10cat',
       'valence_10cat', 'tempo_10cat',  'img_emb', 'track_name_emb'] #'genres_vec',

mus_feat2 = ['tempo_10cat', 'liveness_10cat', 'instrumentalness_10cat', 'speechiness_10cat', 'loudness_10cat', 'acousticness_10cat', 'danceability_10cat', 'valence_10cat', 'energy_10cat']

args_for_pinsage_FAIR_train_LFM = [ 
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.1, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.1, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.1, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.1, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.1, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.2, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.2, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.2, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.2, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.2, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.3, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.3, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.3, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.3, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.3, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.4, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.4, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.4, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.4, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.4, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.5, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.5, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.5, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.5, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.5, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.6, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.6, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.6, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.6, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.6, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.7, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.7, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.7, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.7, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.7, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.8, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.8, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.8, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.8, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.8, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v1_"), gamma=0.9, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v2_"), gamma=0.9, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v3_"), gamma=0.9, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v4_"), gamma=0.9, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),
    ArgsforFairPSRuns(emb=music_embs2, projection_feat=all_feat_set, projection_concat=mus_feat2, hidden_size=2048+512+144, fair_feat_set=mus_feat2, output_path=os.path.join(scratch_dir2, "Unweighted/v5_"), gamma=0.9, alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat'),

    ]









# ArgsforFairPSRuns(emb=[], projection_feat=['img_emb'], projection_concat=[], hidden_size=2048, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img_img")), 
#     ArgsforFairPSRuns(emb=[], projection_feat=['track_name_emb'], projection_concat=[], hidden_size=512, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "track_track")), 
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=['img_emb'], projection_concat=[], hidden_size=144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "mus_dance")), 
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=['img_emb'], projection_concat=[], hidden_size=144, fair_feat_set=['valence'], output_path=os.path.join(scratch_dir2, "mus_valence")), 
#     ArgsforFairPSRuns(emb=[], projection_feat=['img_emb', 'track_name_emb'], projection_concat=[], hidden_size=2048+512, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img+track_img")), 
#     ArgsforFairPSRuns(emb=[], projection_feat=['img_emb', 'track_name_emb'], projection_concat=[], hidden_size=2048+512, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "img+track_track")), 
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "img+track+mus_dance")), 
#     ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"))

'''
Ran: 
ArgsforFairPSRuns(emb=[], projection_feat=['img_emb'], projection_concat=[], hidden_size=2048, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img_img")), 
    ArgsforFairPSRuns(emb=[], projection_feat=['track_name_emb'], projection_concat=[], hidden_size=512, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "track_track")), 
    ArgsforFairPSRuns(emb=music_embs, projection_feat=['img_emb'], projection_concat=[], hidden_size=144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "mus_dance")), 
    ArgsforFairPSRuns(emb=music_embs, projection_feat=['img_emb'], projection_concat=[], hidden_size=144, fair_feat_set=['valence'], output_path=os.path.join(scratch_dir2, "mus_valence")), 
    ArgsforFairPSRuns(emb=[], projection_feat=['img_emb', 'track_name_emb'], projection_concat=[], hidden_size=2048+512, fair_feat_set=['img_emb'], output_path=os.path.join(scratch_dir2, "img+track_img")), 
    ArgsforFairPSRuns(emb=[], projection_feat=['img_emb', 'track_name_emb'], projection_concat=[], hidden_size=2048+512, fair_feat_set=['track_name_emb'], output_path=os.path.join(scratch_dir2, "img+track_track")), 
    ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['danceability'], output_path=os.path.join(scratch_dir2, "img+track+mus_dance")), 
    ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=mus_feat, output_path=os.path.join(scratch_dir2, "img+track+mus_allmus"))

Left to Run: 
ArgsforFairPSRuns(emb=music_embs, projection_feat=mus_feat + ['img_emb', 'track_name_emb'], projection_concat=mus_feat, hidden_size=2048+512+144, fair_feat_set=['valence'], output_path=os.path.join(scratch_dir2, "img+track+mus_valence"))

'''
### Helpers for Launching PinSAGE Model Train ###

feature_groups = {'music': ['tempo', 'liveness', 'instrumentalness', 'speechiness', 'loudness', 'acousticness', 'danceability', 'valence', 'energy'],
                'img': ['img_emb'], 
                'artist': ['followers_10cat', 'popularity_10cat'], 
                'name_embeddings': ['track_name_emb'], 
                'genre': ['genre'],
                'meta': ['tid']}
                # 'meta': ['tid', 'alid', 'arid']}

class ArgsForPSExpRuns(NamedTuple):
    music: bool
    artist: bool
    img: bool
    name_embeddings: bool
    genre: bool
    meta: bool 
    norm: bool


#Generate experiments for largescale hyperparam runs 

def generate_experiment_perms():
    feats = ArgsForPSExpRuns._fields  
    feature_sets = list(product([True, False], repeat=len(feats))) 
    feature_sets.remove((False,False, False, False, False, False, False))
    experiments = []
    for a in feature_sets: 
        experiments.append(ArgsForPSExpRuns(**dict(zip(feats, a))))         
    return experiments 

# args_for_pinsage_train = generate_experiment_perms()

#Manual experiment entry
args_for_pinsage_train = [ 
    ArgsForPSExpRuns(music=False, artist=False, img=False, genre= False, name_embeddings=False, meta=True, norm=False),     
]


def prepare_for_config(args): 
    all_features = [] 
    concat_features = [] 
    categories = [] 
    if args.music: 
        categories.append('music')
        all_features.extend(feature_groups['music'])
        concat_features.extend(feature_groups['music'])
    if args.artist: 
        categories.append('artist')
        all_features.extend(feature_groups['artist'])
        concat_features.extend(feature_groups['artist'])
    if args.genre: 
        categories.append('genre')
        all_features.extend(feature_groups['genre'])
        concat_features.extend(feature_groups['genre'])
    if args.name_embeddings: 
        categories.append('track')
        all_features.extend(feature_groups['name_embeddings'])
    if args.img: 
        categories.append('album')
        all_features.extend(feature_groups['img'])
    if args.meta: 
        categories.append('meta')
        all_features.extend(feature_groups['meta'])
        concat_features.extend(feature_groups['meta'])

    output_path = os.path.join(scratch_dir, "_".join(categories))  
    return all_features, concat_features, output_path

# def find_latest_checkpoint(output_path): 
#     checkpoints = glob.glob(output_path+"checkpoints/*")
#     return sorted(checkpoints)[-1]

### Helpers for Building Track Embeddings from Trained Model ###

class ArgsForPSRecs(NamedTuple):  
    test_set: str 
    k: int
    gen_amount: int 
params = { 
    'test_set': ['TS1'],
    'k': [10000], 
    'gen_amount': [10], 
    
    }

# args_for_pinsage_rec = [ArgsForPSRecs(**dict(zip(params, v))) for v in product(*params.values())]

# def generate_configs(tuple, params): 
#     return [tuple(**dict(zip(params, v))) for v in product(*params.values())]


# args_for_pinsage_rec = [
#     #individual feature sets 
#     ArgsForPSRecs(track_embed_path = scratch_dir +'music_artist_genre_track_album_meta_FOCAL_LOSS/track_emb/embeddings_as_array_fullg.pkl', test_set = test_set_paths[0], output_path=scratch_dir +'music_artist_genre_track_album_meta_FOCAL_LOSS/recs_TS1/', k = 500, gen_amount = 10)
# ] 

