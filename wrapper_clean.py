import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

#My code 
from src2.utils.parsers.final_parser import *  
from src2.test_pipeline import *
from src2.fair_train_pipeline import * 
from src2.utils.misc import * 
from src2.utils.save_res import * 
from src2.eval.gen_embeddings import * 
from src2.eval.gen_recommendations import * 

#Benchmarks 
from src2.benchmarks.ScoreReg.parser import * 
from src2.benchmarks.ScoreReg.main_graph import * 
from src2.benchmarks.XQuaD.xquad_main import launch_xqad_run 

#Configs 

#Libraries 
import numpy as np 
import sys 
import os 
import time 
import torch 
import pickle 
import glob 
from tqdm import tqdm
import argparse 
import json 
# import ipdb

def launch_redress(runs):
    #Get task number 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    jobid = os.environ.get("SLURM_ARRAY_JOB_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path, args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_G_{}_A_{}_B_{}/'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.gamma, run_args.alpha, run_args.boost)
    
    if exp_name == 'BOOST': 
        exp_path = f'{base_path}/{dataset_args.name}/{exp_name}/{run_args.method}/{run_args.version}_G_{run_args.gamma}_A_{run_args.alpha}/'
        if not os.path.exists(f'{base_path}/{dataset_args.name}/{exp_name}/{run_args.method}/'): 
            os.mkdir(f'{base_path}/{dataset_args.name}/{exp_name}/{run_args.method}/')
       
    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()

    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()

    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    cfg.DATASET.NAME = 'NO_ISOLATE' 
    cfg.TRAIN.SOLVER.BASE_LR = 0.0001
    cfg.TRAIN.SOLVER.DECAY = False 
    cfg.TRAIN.LOSS = 'FOCAL_LOSS'
    cfg.MODEL.PINSAGE.PROJECTION.NORMALIZE  = False
    cfg.MODEL.PINSAGE.REPRESENTATION_NORMALIZE = False
   
    cfg.TRAIN.UTILITY_EPOCHS = 20
    cfg.TRAIN.FAIR_EPOCHS = 15

    cfg.seed = run_args.seed
    cfg.MODEL.PINSAGE.DROPOUT = run_args.dropout 
    cfg.MODEL.PINSAGE.PROJECTION.EMB = run_args.emb 
    cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES = run_args.projection_feat 
    cfg.MODEL.PINSAGE.PROJECTION.CONCAT = run_args.projection_concat 
    cfg.MODEL.PINSAGE.HIDDEN_SIZE = run_args.hidden_size 
    cfg.FAIR.FEAT_SET = run_args.fair_feat_set 
    cfg.FAIR.FAIRNESS_BALANCE = run_args.gamma
    cfg.FAIR.ALPHA = run_args.alpha
    cfg.FAIR.BOOST = run_args.boost
    cfg.FAIR.NDCG_METHOD = run_args.method
    cfg.FAIR.POP_FEAT = run_args.pop_feat
    cfg.OUTPUT_PATH = exp_path
    
    checkpoint_path, u_epoch = find_latest_checkpoint_clean(exp_path, mode='u')
    f_checkpoint_path, uf_epoch = find_latest_checkpoint_clean(exp_path, mode='u+f')
    
    if u_epoch <  (cfg.TRAIN.UTILITY_EPOCHS -1)  or uf_epoch < (cfg.TRAIN.UTILITY_EPOCHS + cfg.TRAIN.FAIR_EPOCHS-1) : 
        fair_train_main(cfg) 
         

    with open(exp_path + "run_args.pkl", "w") as f:
        f.write(json.dumps(args_for_function._asdict()))

    utility_path = '{}{}/'.format(exp_path, 'utility')
    fair_path = '{}{}/'.format(exp_path, 'redress')

    #Loading Utility Embeddings 
    u_track_path = '{}{}'.format(utility_path, 'u_track_emb.pkl')
    if len(glob.glob(u_track_path)) == 0:  
        checkpoint_path, u_epoch = find_latest_checkpoint_clean(exp_path, mode='u')
        track_embeddings, u_track_path = gen_track_embeddings_clean(cfg, checkpoint_path, output_path = utility_path)

    #Loading Fair Embeddings 
    uf_track_path = '{}{}'.format(fair_path, 'u_track_emb.pkl')
    if len(glob.glob(uf_track_path)) == 0: 
        checkpoint_path, uf_epoch = find_latest_checkpoint_clean(exp_path, mode='u+f')
        track_embeddings, uf_track_path =  gen_track_embeddings_clean(cfg, checkpoint_path, output_path = fair_path)

    u_validation_path = f'{utility_path}val/'
    f_validation_path = f'{fair_path}val/'

    u_test_path = f'{utility_path}test/'
    f_test_path = f'{fair_path}test/'

    if not os.path.exists(u_validation_path): 
        os.mkdir(u_validation_path)
    if not os.path.exists(f_validation_path): 
        os.mkdir(f_validation_path)
    if not os.path.exists(u_test_path): 
        os.mkdir(u_test_path)
    if not os.path.exists(f_test_path): 
        os.mkdir(f_test_path)

    
    #Loading Recommendations -- VALID 
    # Utility 
    u_rec_path = '{}{}'.format(u_validation_path, 'u_rec_tracks.pkl')
    if len(glob.glob(u_rec_path)) == 0: 
        u_track_path = '{}{}'.format(utility_path, 'u_track_emb.pkl')
        u_rec_df, u_rec_path = gen_recommendations_cosine_clean(cfg, k=500, gen_amount=10, track_embed_path= u_track_path , output_path=u_validation_path, mode = 'valid') 

    #Fair
    uf_rec_path = '{}{}'.format(f_validation_path, 'u_rec_tracks.pkl')
    if len(glob.glob(uf_rec_path)) == 0: 
        uf_track_path = '{}{}'.format(fair_path, 'u_track_emb.pkl')
        uf_rec_df, uf_rec_path = gen_recommendations_cosine_clean(cfg, k=500, gen_amount=10, track_embed_path=uf_track_path , output_path=f_validation_path, mode = 'valid') 
    
    
    #Evaluate -- VALID 
    u_rec_path = '{}{}'.format(u_validation_path, 'u_rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = uf_epoch,recommended_track_path = uf_rec_path,  verbose=True, output_path=u_validation_path, mode = 'valid')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch, recommended_track_path=uf_rec_path, output_path=u_validation_path, mode='PS', setting = 'valid')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch,recommended_track_path=uf_rec_path, output_path=u_validation_path, mode='PS', setting = 'valid')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch,recommended_track_path=uf_rec_path, output_path=u_validation_path, mode='PS', setting = 'valid')

    uf_rec_path = '{}{}'.format(f_validation_path, 'u_rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = uf_epoch,recommended_track_path = uf_rec_path,  verbose=True, output_path=f_validation_path, mode = 'valid')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch, recommended_track_path=uf_rec_path, output_path=f_validation_path, mode='PS', setting = 'valid')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch,recommended_track_path=uf_rec_path, output_path=f_validation_path, mode='PS', setting = 'valid')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = uf_epoch,recommended_track_path=uf_rec_path, output_path=f_validation_path, mode='PS', setting = 'valid')

    #Loading Recommendation -- TEST
    #utility
    u_rec_path = '{}{}'.format(u_test_path, 'u_rec_tracks.pkl')
    if len(glob.glob(u_rec_path)) == 0: 
        u_track_path = '{}{}'.format(utility_path, 'u_track_emb.pkl')
        u_rec_df, u_rec_path = gen_recommendations_cosine_clean(cfg, k=500, gen_amount=10, track_embed_path= u_track_path , output_path=u_test_path, mode = 'test') 

    #Fair
    uf_rec_path = '{}{}'.format(f_test_path, 'u_rec_tracks.pkl')
    if len(glob.glob(uf_rec_path)) == 0: 
        uf_track_path = '{}{}'.format(fair_path, 'u_track_emb.pkl')
        uf_rec_df, uf_rec_path = gen_recommendations_cosine_clean(cfg, k=500, gen_amount=10, track_embed_path=uf_track_path , output_path=f_test_path, mode = 'test') 
    
    #Evaluate on Test 
    #Utility
    u_rec_path = '{}{}'.format(u_test_path, 'u_rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = u_epoch,recommended_track_path = u_rec_path,  verbose=True, output_path=u_test_path)
    cfg.FAIR.POP_FEAT = 'appear_pop'
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=u_rec_path, output_path=u_test_path, mode='PS')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=u_rec_path, output_path=u_test_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=u_rec_path, output_path=u_test_path, mode='PS')

    #Fairness -- TEST
    uf_rec_path = '{}{}'.format(f_test_path, 'u_rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = u_epoch,recommended_track_path = uf_rec_path,  verbose=True, output_path=f_test_path)
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch, recommended_track_path=uf_rec_path, output_path=f_test_path, mode='PS')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=uf_rec_path, output_path=f_test_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=uf_rec_path, output_path=f_test_path, mode='PS')
 
def launch_scorereg(runs):
    # scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS'  
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_{}_G_{}/'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.model, run_args.weight)
    run_args = run_args._replace(output_path = exp_path)
    print("EXP_PATH", exp_path)

    dataset_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data_Final2/'

    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()
    data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount2'

    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()
        data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount2'

    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    
    checkpoint_path, last_epoch = load_ScoreReg_checkpoints(exp_path)
    if last_epoch < 19 and last_epoch > 0: 
        print("loaded checkpoint of epoch", last_epoch, checkpoint_path)
        train(run_args, data_path, pretrained_path = checkpoint_path, epoch_skip = last_epoch)
    if last_epoch == -1: 
        print("no checkpoints found, starting from scratch")
        train(run_args, data_path) 

    with open(exp_path + "run_args.pkl", "w") as f:
        f.write(json.dumps(args_for_function._asdict()))

    model_path = exp_path + 'checkpoints/model19.pth'
    emb_path = exp_path + "PS_track_emb.pkl"

    OG_path = f'{exp_path}OG/'
    if not os.path.exists(OG_path): 
        os.mkdir(OG_path)
    PS_path = f'{exp_path}PS/'
    if not os.path.exists(PS_path): 
        os.mkdir(PS_path)

    val_path = f'{OG_path}val/'
    test_path = f'{OG_path}test/'
    if not os.path.exists(val_path): 
        os.mkdir(val_path)
    if not os.path.exists(test_path): 
        os.mkdir(test_path)

    #REC GEN PS  
    print("*** Generating recs***")
    
    #REC GEN OG
    #Valid
    make_recs_OG(cfg, model_path=model_path, output_path=val_path, data_path=data_path, model_name=run_args.model, rec_num=100, mode = 'valid')
    rec_path = '{}{}'.format(val_path, 'rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=val_path, mode='valid')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    
    #Test
    make_recs_OG(cfg, model_path=model_path, output_path=test_path, data_path=data_path, model_name=run_args.model, rec_num=100)
    rec_path = '{}{}'.format(test_path, 'rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=test_path)
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS', )
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS', )
   
def launch_xquad(runs):
    scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS' 
    #Get task number 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_G_{}/'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.gamma)
    if not os.path.exists(exp_path): 
        os.mkdir(exp_path)
    with open(exp_path + "run_args.pkl", "w") as f:
        f.write(json.dumps(args_for_function._asdict()))

    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()

    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()

    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    cfg.FAIR.POP_FEAT = run_args.pop_feat
    cfg.OUTPUT_PATH = exp_path 
    REDRESS_path = '{}/{}/{}/{}_G_{}_A_{}_B_{}/{}/{}'.format(scratch_path, dataset_args.name, 'REDRESS', run_args.version, run_args.gamma, run_args.alpha, run_args.boost, 'utility', 'u_rec_tracks.pkl')

    print(f"loading reclist from REDRESS_path:{REDRESS_path}")

    output_path = '{}{}'.format(exp_path, 'xquad_recs.pkl')
    
    launch_xqad_run(p_rec=REDRESS_path, p_gt=cfg.DATASET.TEST_DATA_PATH, p_data=cfg.DATASET.DATA_PATH, output_path = output_path, lmbda=run_args.gamma, LT_col =run_args.pop_feat)  
    
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = "NONE", recommended_track_path = output_path,  verbose=True, output_path=exp_path)
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = "NONE", recommended_track_path=output_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = "NONE", recommended_track_path=output_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = "NONE", recommended_track_path=output_path, output_path=exp_path, mode='PS')
      
def launch_macr(runs):  
    # scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS'  
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_{}_MACR/'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.model, run_args.weight)
    run_args = run_args._replace(output_path = exp_path)

    
    dataset_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data_Final2/'

    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()
    data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount2'

    
    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()
        data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount2'

    
    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    
    checkpoint_path, last_epoch = load_ScoreReg_checkpoints(exp_path)
    if last_epoch < 19 and last_epoch > 0: 
        print("loaded checkpoint of epoch", last_epoch, checkpoint_path)
        train(run_args, data_path, pretrained_path = checkpoint_path, epoch_skip = last_epoch)
    if last_epoch < 0: 
        print("no checkpoints found, starting from scratch")
        train(run_args, data_path) 

    with open(exp_path + "run_args.pkl", "w") as f:
        f.write(json.dumps(args_for_function._asdict()))


    model_path = exp_path + 'checkpoints/model19.pth'
    
    OG_path = f'{exp_path}OG/'
    if not os.path.exists(OG_path): 
        os.mkdir(OG_path)
    
    test_path = f'{OG_path}test/'
    
    if not os.path.exists(test_path): 
        os.mkdir(test_path)

    #REC GEN OG
    for c in [5, 10, 20, 25, 30, 35, 40, 45]:
        val_path = f'{OG_path}val_{c}/'
        if not os.path.exists(val_path): 
            os.mkdir(val_path) 
        make_recs_OG(cfg, model_path=model_path, output_path=val_path, data_path=data_path, model_name='MACR', rec_num=100, c=c, mode='valid')
        rec_path = '{}{}'.format(val_path, f'rec_tracks_{c}.pkl')
        launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=val_path, mode='valid')
        cfg.FAIR.POP_FEAT = '80_20_LT'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS')
        cfg.FAIR.POP_FEAT = 'log10_popcat'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS')
        cfg.FAIR.POP_FEAT = 'appear_pop'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS')
    
    for c in [5, 10, 20, 25, 30, 35, 40, 45]:
        test_path = f'{OG_path}test_{c}/'
        if not os.path.exists(test_path): 
            os.mkdir(test_path) 
        make_recs_OG(cfg, model_path=model_path, output_path=test_path, data_path=data_path, model_name='MACR', rec_num=100, c=c, mode='test')
        rec_path = '{}{}'.format(test_path, f'rec_tracks_{c}.pkl')
        launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=test_path)
        cfg.FAIR.POP_FEAT = '80_20_LT'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS')
        cfg.FAIR.POP_FEAT = 'log10_popcat'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS')
        cfg.FAIR.POP_FEAT = 'appear_pop'
        launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS')
     
def launch_vanilla_pinsage():  
    #Get task number 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = PS_runs_LFM_Filtered[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}/'.format(base_path, dataset_args.name, exp_name, run_args.version, )
    
    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()

    if dataset_args.name == 'MPD': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()

    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    cfg.DATASET.NAME = 'NO_ISOLATE' 
    cfg.TRAIN.SOLVER.BASE_LR = 0.0001
    cfg.TRAIN.SOLVER.DECAY = False 
    cfg.TRAIN.LOSS = 'FOCAL_LOSS'
    cfg.MODEL.PINSAGE.PROJECTION.NORMALIZE  = False
    cfg.MODEL.PINSAGE.REPRESENTATION_NORMALIZE = False
    cfg.MODEL.PINSAGE.DROPOUT = 0.0

    cfg.TRAIN.UTILITY_EPOCHS = 40
    cfg.TRAIN.FAIR_EPOCHS = 0
    cfg.EARLY_STOPPING = True
    
    cfg.MODEL.PINSAGE.PROJECTION.EMB = run_args.emb 
    cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES = run_args.projection_feat 
    cfg.MODEL.PINSAGE.PROJECTION.CONCAT = run_args.projection_concat 
    cfg.MODEL.PINSAGE.HIDDEN_SIZE = run_args.hidden_size 
    cfg.FAIR.FEAT_SET = run_args.fair_feat_set 
    cfg.FAIR.FAIRNESS_BALANCE = run_args.gamma
    cfg.FAIR.POP_FEAT = run_args.pop_feat
    cfg.FAIR.NDCG_METHOD = run_args.method 
    cfg.OUTPUT_PATH = exp_path
    
    checkpoint_path, u_epoch = find_latest_checkpoint_clean(exp_path, mode='u')

   
    #Loading Utility Embeddings 
    u_track_path = '{}{}'.format(exp_path, 'u_track_emb.pkl')
    if len(glob.glob(u_track_path)) == 0:  
        checkpoint_path, u_epoch = find_latest_checkpoint_clean(exp_path, mode='u')
        track_embeddings, u_track_path = gen_track_embeddings_clean(cfg, checkpoint_path, output_path = exp_path)

    #Loading Utility Recommendations 
    u_output_path = '{}{}'.format(exp_path, 'u_rec_tracks.pkl')
    if len(glob.glob(u_rec_path)) == 0: 
        u_track_path = '{}{}'.format(exp_path, 'u_track_emb.pkl')
        u_rec_df, u_rec_path = gen_recommendations_cosine_clean(cfg, k=500, gen_amount=10, track_embed_path= u_track_path , output_path=exp_path) 
        
    #Evaluate 
    u_rec_path = '{}{}'.format(exp_path, 'u_rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = u_epoch,recommended_track_path = u_rec_path,  verbose=True, output_path=exp_path)
    launch_fairness_audit_clean(cfg, k=100, epoch = u_epoch,recommended_track_path=u_rec_path, output_path=exp_path, mode='PS')

def launch_vanilla_LGCN(runs): 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_{}/'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.model)
    run_args = run_args._replace(output_path = exp_path)

    if exp_name == 'LGCN_EMB': 
        exp_path = f'{base_path}/{dataset_args.name}/LGCN_EMB/{run_args.emb_size}/{run_args.version}_{run_args.model}/'
        if not os.path.exists(f'{base_path}/{dataset_args.name}/LGCN_EMB/{run_args.emb_size}/'): 
            os.mkdir(f'{base_path}/{dataset_args.name}/LGCN_EMB/{run_args.emb_size}/')
        if not os.path.exists(f'{base_path}/{dataset_args.name}/LGCN_EMB/{run_args.emb_size}/{run_args.version}_{run_args.model}/'): 
            os.mkdir(f'{base_path}/{dataset_args.name}/LGCN_EMB/{run_args.emb_size}/{run_args.version}_{run_args.model}/')
        run_args = run_args._replace(output_path = exp_path)
    
    if exp_name == 'LGCN_LR': 
        exp_path = f'{base_path}/{dataset_args.name}/LGCN_LR/{run_args.lr}/{run_args.version}_{run_args.model}/'
        if not os.path.exists(f'{base_path}/{dataset_args.name}/LGCN_LR/{run_args.lr}/'): 
            os.mkdir(f'{base_path}/{dataset_args.name}/LGCN_LR/{run_args.lr}/')
        if not os.path.exists(f'{base_path}/{dataset_args.name}/LGCN_LR/{run_args.lr}/{run_args.version}_{run_args.model}/'): 
            os.mkdir(f'{base_path}/{dataset_args.name}/LGCN_LR/{run_args.lr}/{run_args.version}_{run_args.model}/')
        run_args = run_args._replace(output_path = exp_path)
    


    print(f"EXP_PATH:{exp_path}")
    dataset_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data_Final2/'

    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()
    data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount'

    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()
        data_path = f'{dataset_path}{dataset_args.name}/ScoreRegWithGenAmount2'

    
    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    
    checkpoint_path, last_epoch = load_ScoreReg_checkpoints(exp_path)
    
    print(f"last epoch:{last_epoch}")
    if last_epoch < 19 and last_epoch > 0: 
        print("loaded checkpoint of epoch", last_epoch, checkpoint_path)
        train(run_args, data_path, pretrained_path = checkpoint_path, epoch_skip = last_epoch)
    
    if last_epoch == -1: 
        print("no checkpoints found, starting from scratch")
        train(run_args, data_path) 
    
    with open(exp_path + "run_args.pkl", "w") as f:
        f.write(json.dumps(args_for_function._asdict()))
    OG_path = f'{exp_path}OG/'
    if not os.path.exists(OG_path): 
        os.mkdir(OG_path)
    
    val_path = f'{OG_path}val/'
    test_path = f'{OG_path}test/'
    if not os.path.exists(val_path): 
        os.mkdir(val_path)
    if not os.path.exists(test_path): 
        os.mkdir(test_path)

    model_path = exp_path + 'checkpoints/model19.pth'
    emb_path = exp_path + "PS_track_emb.pkl"

    # gen_rec = 
    

    make_recs_OG(cfg, model_path=model_path, output_path=val_path, data_path=data_path, model_name=run_args.model, rec_num=100, mode = 'valid')
    rec_path = '{}{}'.format(val_path, 'rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=val_path, mode='valid')
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=val_path, mode='PS', setting='valid')
    
    #Test
    make_recs_OG(cfg, model_path=model_path, output_path=test_path, data_path=data_path, model_name=run_args.model, rec_num=100)
    rec_path = '{}{}'.format(test_path, 'rec_tracks.pkl')
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = run_args.epochs, recommended_track_path = rec_path,  verbose=True, output_path=test_path)
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS', )
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = run_args.epochs, recommended_track_path=rec_path, output_path=test_path, mode='PS', )
   
def launch_popularity_benchmark(runs): 
    scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS' 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = runs[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params

    exp_path = '{}/{}/{}/'.format(scratch_path, dataset_args.name, "POP")
    if not os.path.exists(exp_path): 
        os.mkdir(exp_path)

    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()
    
    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()
        
    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path
    

    train_set = pickle.load(open( cfg.DATASET.DATA_PATH, "rb"))
    test_set = pickle.load(open( cfg.DATASET.TEST_DATA_PATH, "rb"))
    top_tracks = train_set['df_track'].sort_values('appear_raw', ascending=False).iloc[:100, :].tid.to_list() 
    rec_df = pd.DataFrame({'pid': test_set.pid.unique(), 'recs': [top_tracks] * len(test_set.pid.unique())})
    
    save_path = exp_path + 'pop_recs.pkl'
    pickle.dump(rec_df, open(save_path, "wb"))

    launch_performance_eval_clean(cfg, gen_amount=0, epoch = "N/A", recommended_track_path = save_path,  verbose=True, output_path=exp_path)
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=save_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=save_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=save_path, output_path=exp_path, mode='PS')

def launch_musfeat_benchmark(runs):
    args_for_function = runs[0] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path, exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/BARE_FEAT/'.format(base_path, dataset_args.name)
    
    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()

    if dataset_args.name == 'MPD_Subset': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()

    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path

    cfg.MODEL.PINSAGE.PROJECTION.EMB = run_args.emb 
    cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES = run_args.projection_feat 
    cfg.MODEL.PINSAGE.PROJECTION.CONCAT = run_args.projection_concat 
    cfg.MODEL.PINSAGE.HIDDEN_SIZE = run_args.hidden_size 

    
    rec_path = '{}{}'.format(exp_path, 'raw_feat_recs.pkl')
    
    recommendations = gen_baseline_recommendations_all(cfg, name_embeddings = True, music=True, artist = False, img=True, genre=False, k=100, scale=False, bin=False, gen_amount=10, output_path=rec_path)
    
    launch_performance_eval_clean(cfg, gen_amount=0, epoch = "N/A", recommended_track_path = rec_path,  verbose=True, output_path=exp_path)
    cfg.FAIR.POP_FEAT = '80_20_LT'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=rec_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'log10_popcat'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=rec_path, output_path=exp_path, mode='PS')
    cfg.FAIR.POP_FEAT = 'appear_pop'
    launch_fairness_audit_clean(cfg, k=100, epoch = "N/A", recommended_track_path=rec_path, output_path=exp_path, mode='PS')

def launch_random_baseline(): 
    scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS' 
    #Get task number 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    print(task_id, type(task_id))
    args_for_function = REDRESS_runs_LFM_Filtered[int(task_id)] 
    print("RUNNING WITH ARGS:{}".format(args_for_function)) 
    base_path,exp_name, dataset_args, run_args = args_for_function.base_path,args_for_function.exp_name, args_for_function.dataset, args_for_function.run_params
    exp_path = '{}/{}/{}/{}_G_{}_A_{}_B_{}'.format(base_path, dataset_args.name, exp_name, run_args.version, run_args.gamma, run_args.alpha, run_args.boost)
    
    from src2.utils.config.LastFM import get_cfg_defaults
    cfg = get_cfg_defaults()

    if dataset_args.name == 'MPD': 
        from src2.utils.config.full import get_cfg_defaults
        cfg = get_cfg_defaults()


    cfg.DATASET.DATA_PATH = dataset_args.train_path
    cfg.DATASET.TEST_DATA_PATH = dataset_args.test_path 

    print(exp_path)
    
    checkpoint_path, u_epoch = find_latest_checkpoint_clean(exp_path, mode='u')
    
    utility_path = '{}/{}/'.format(exp_path, 'utility')

    #Loading Utility Embeddings 
    u_track_path = '{}{}'.format(utility_path, 'u_track_emb.pkl')
    
    u_rec_path = '{}{}'.format(utility_path, 'u_rec_tracks.pkl')
    if len(glob.glob(u_rec_path)) == 0: 
        u_track_path = '{}{}'.format(utility_path, 'u_track_emb.pkl')
        u_rec_df, u_rec_path = gen_recommendations_random(cfg, k=500, gen_amount=10, track_embed_path= u_track_path , output_path=exp_path)    



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
        type=str)
    parser.add_argument("--dataset", 
        type=str)
    args = parser.parse_args()

    if args.mode == 'REDRESS': 
        if args.dataset == 'MPD': 
            launch_redress(REDRESS_runs_MPD)
        if args.dataset == 'LFM': 
            launch_redress(REDRESS_runs_LFM)

    if args.mode == 'SR': 
        if args.dataset == 'MPD': 
            launch_scorereg(SR_runs_MPD)
        if args.dataset == 'LFM': 
            launch_scorereg(SR_runs_LFM)
    
    if args.mode == 'XQUAD': 
        if args.dataset == 'MPD': 
            launch_xquad(XQUAD_runs_MPD)
        if args.dataset == 'LFM': 
            launch_xquad(XQUAD_runs_LFM)

    if args.mode == 'MACR': 
        if args.dataset == 'MPD': 
            launch_macr(MACR_runs_MPD) 
        if args.dataset == 'LFM': 
            launch_macr(MACR_runs_LFM) 
    
    if args.mode == 'LGCN': 
        if args.dataset == 'MPD': 
            launch_vanilla_LGCN(LGCN_runs_MPD)
        if args.dataset == 'LFM': 
            launch_vanilla_LGCN(LGCN_runs_LFM)

    if args.mode == 'BOOST': 
        if args.dataset == 'MPD': 
            launch_redress(BOOST_hp_runs_MPD) 
        if args.dataset == 'LFM': 
            launch_redress(BOOST_hp_runs_LFM) 
    
    if args.mode == 'POP': 
        if args.dataset == 'MPD': 
            launch_popularity_benchmark(REDRESS_runs_MPD)
        if args.dataset == 'LFM': 
            launch_popularity_benchmark(REDRESS_runs_LFM)

    if args.mode == 'BARE_FEAT': 
        if args.dataset == 'MPD': 
            launch_musfeat_benchmark(REDRESS_runs_MPD)
        if args.dataset == 'LFM': 
            launch_musfeat_benchmark(REDRESS_runs_LFM)

    if args.mode == 'MACR_VAL': 
        if args.dataset == 'MPD': 
            validate_MACR(MACR_runs_MPD)

    if args.mode == 'EXP_BOOST': 
        if args.dataset == 'MPD': 
            launch_redress(BOOST_hp_runs_MPD) 
        if args.dataset == 'LFM': 
            launch_redress(BOOST_hp_runs_LFM)  
    
    if args.mode == 'LGCN_EMB': 
        if args.dataset == 'MPD': 
            print("NOPE")
        if args.dataset == 'LFM': 
            launch_vanilla_LGCN(emb_size_hp_runs_LFM)

