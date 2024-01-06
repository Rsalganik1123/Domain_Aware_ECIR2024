
import argparse
from typing import NamedTuple
import os
from functools import partial

test_set1 = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set1.pkl'
test_set2 = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/test_set2.pkl'
output_path1 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/track_baseline_TS1_scaled.pkl'
output_path2 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/track_baseline_TS2_scaled.pkl'
output_path3 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/img_baseline_TS1_scaled.pkl'
output_path4 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/img_baseline_TS2_scaled.pkl'
output_path5 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/genre_baseline_TS1_scaled.pkl'
output_path6 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/genre_baseline_TS2_scaled.pkl'
output_path7 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/artist_baseline_TS1_scaled.pkl'
output_path8 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/artist_baseline_TS2_scaled.pkl'
output_path9 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/music_baseline_TS1_scaled.pkl'
output_path10 = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs2/music_baseline_TS2_scaled.pkl'

baseline_folder = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_recs/'
results_folder = '/home/mila/r/rebecca.salganik/Projects/MusicSAGE/temp_results/baseline_rec_eval/'

class ArgsForBaseLineRuns(NamedTuple):
    music: bool
    artist: bool
    img: bool
    name_embeddings: bool
    genre: bool
    scale: bool 
    test_set: str
    output_path: str
    gen_amount: int
    k: int 

args_for_baseline_exp = [
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=False, genre= False, name_embeddings=True, scale = True, test_set =test_set1, output_path=output_path1),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=False,  genre= False, name_embeddings=True, scale = True, test_set =test_set2, output_path=output_path2),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=True,  genre= False, name_embeddings=False, scale = True, test_set =test_set1, output_path=output_path3),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=True,  genre= False, name_embeddings=False, scale = True, test_set =test_set2, output_path=output_path4),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=False,  genre= True, name_embeddings=False, scale = True, test_set =test_set1, output_path=output_path5),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=False,  genre= True, name_embeddings=False, scale = True, test_set =test_set2, output_path=output_path6),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=False, img=False,  genre= False, name_embeddings=False, scale = True, test_set =test_set1, output_path=output_path7),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=True, img=False,  genre= False, name_embeddings=False, scale = True, test_set =test_set2, output_path=output_path8),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=False, artist=True, img=False,  genre= False, name_embeddings=False, scale = True, test_set =test_set1, output_path=output_path9),
    ArgsForBaseLineRuns(gen_amount=20, k = 500, music=True, artist=True, img=False,  genre= False, name_embeddings=False, scale = True, test_set =test_set2, output_path=output_path10),
]

class ArgsForBaseLineEval(NamedTuple):
    gen_amount: int
    k: int 
    recommended_track_path: str 
    test_set: str 
    output_path: str 

args_for_baseline_eval = [
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path1, test_set=test_set1, output_path=results_folder), 
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path2, test_set=test_set2, output_path=results_folder), 
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path3, test_set=test_set1, output_path=results_folder), 
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path4, test_set=test_set2, output_path=results_folder), 
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path5, test_set=test_set1, output_path=results_folder), 
    ArgsForBaseLineEval(gen_amount=20, k = 500, recommended_track_path=output_path6, test_set=test_set2, output_path=results_folder)
]
