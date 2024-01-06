# from src2.utils.config.full import get_cfg_defaults
# from src2.model.build import build_model
# from src2.graph_build.data_load import build_dataset

from src2.model.build import build_model
# from src2.graph_build.data_load import build_dataset

from src2.graph_build.data_load import build_dataset
import src2.graph_build.spotify_dataset
from torch.utils.data import IterableDataset, DataLoader
from src2.sampler.graph_sampler import build_graph_sampler 
from torch.nn.functional import cosine_similarity
import torch 
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm 
import os 
# import ipdb

'''
File contains functions used for generating track embeddings from PinSAGE output 
'''

def gen_track_embeddings_small(cfg, output_path=None): 
    '''
    INPUT: 
        cfg: dictionary with all parameters for general PinSAGE run
        output_path: path for saving track embeddings 
    OUTPUT:
        all_features_cpu: numpy array containing all the feature embeddings 
    '''
    #LOAD CONFIG 
    g, train_g, [train_user_ids, val_user_ids, test_user_ids] = build_dataset(cfg)
    #BUILD MODEL / LOAD CHECKPOINTS 
    model = build_model(g, cfg)
    model_state = torch.load(cfg.EMBEDS.CHECKPOINT, map_location='cpu')
    model.load_state_dict(model_state['model_state'])
    
    neighbor_sampler, collator = build_graph_sampler(train_g, cfg)
    dataloader_test = DataLoader(
            torch.arange(g.number_of_nodes('track')),
            batch_size=32,
            collate_fn=collator.collate_test,
            num_workers=1)

    model = model.eval()
    model = model.cuda()
    device = torch.device('cuda:0')

    #GENERATE REPRESENTATIONS 
    all_features_cpu = [] 
    dataloader_it = iter(dataloader_test)
    idx = 0 
    for blocks in tqdm(dataloader_it):
        idx +=1 
        with torch.no_grad():
            for i in range(len(blocks)): 
                blocks[i] = blocks[i].to(device)
            features = model.get_repr(blocks)
            all_features_cpu.append(features.cpu().numpy()) 
    all_features_cpu = np.concatenate(all_features_cpu)
    #SAVE
    if output_path:
        print("***Saving Track Embeddings to {}***".format(output_path))
        pickle.dump(all_features_cpu, open(output_path, "wb"))
    return all_features_cpu

def gen_track_embeddings_clean(cfg,  checkpoint_path, output_path=None): 
    '''
    INPUT: 
        cfg: dictionary with all parameters for general PinSAGE run
        output_path: path for saving track embeddings 
    OUTPUT:
        all_features_cpu: numpy array containing all the feature embeddings 
    '''

    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)   
    torch.cuda.manual_seed(47)
    np.random.seed(47)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #LOAD CONFIG 
    g, train_g, val_g = build_dataset(cfg)
    # assert cfg.DATASET.NAME == 'ENTIRE'
    #BUILD MODEL / LOAD CHECKPOINTS 
    model = build_model(g, cfg)
    model_state = torch.load(checkpoint_path, map_location='cpu')
    print("***Loaded Checkpoint from:{}***".format(checkpoint_path))
    model.load_state_dict(model_state['model_state'])
    
    neighbor_sampler, collator = build_graph_sampler(g, cfg) #should it be train_g or g? 
    
    dataloader_test = DataLoader(
            torch.arange(g.number_of_nodes('track')),
            batch_size=32,
            collate_fn=collator.collate_test,
            num_workers=0)

    model.eval()
    model = model.cuda()
    device = torch.device('cuda:0')
    
    #GENERATE REPRESENTATIONS 
    all_features_cpu = [] 
    all_ids = [] 
    dataloader_it = iter(dataloader_test)
    idx = 0 
    for blocks in tqdm(dataloader_it):
        idx +=1 
        with torch.no_grad():
            for i in range(len(blocks)): 
                blocks[i] = blocks[i].to(device)
            features = model.get_repr(blocks)
            all_features_cpu.append(features.cpu().numpy()) 
            # all_ids.append(blocks[1].dstnodes['track'].data['id'])
    all_features_array = np.concatenate(all_features_cpu)
    #SAVE
    if output_path:
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,'u_track_emb.pkl')
        print("***Saving Track Embeddings to {}***".format(file_path))
        pickle.dump(all_features_array, open(file_path, "wb"))
    return all_features_array, file_path



def gen_track_embeddings(cfg,  checkpoint_path, mode = 'fullg', output_path=None): 
    '''
    INPUT: 
        cfg: dictionary with all parameters for general PinSAGE run
        output_path: path for saving track embeddings 
    OUTPUT:
        all_features_cpu: numpy array containing all the feature embeddings 
    '''

    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)   
    torch.cuda.manual_seed(47)
    np.random.seed(47)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #LOAD CONFIG 
    g, train_g, val_g = build_dataset(cfg)
    # assert cfg.DATASET.NAME == 'ENTIRE'
    #BUILD MODEL / LOAD CHECKPOINTS 
    model = build_model(g, cfg)
    model_state = torch.load(checkpoint_path, map_location='cpu')
    print("***Loaded Checkpoint from:{}***".format(checkpoint_path))
    model.load_state_dict(model_state['model_state'])
    print("mode is: {}".format(mode))
    if mode == 'fullg': 
        neighbor_sampler, collator = build_graph_sampler(g, cfg) #should it be train_g or g? 
    elif mode == 'traing': 
        neighbor_sampler, collator = build_graph_sampler(train_g, cfg)
    else: 
        print("ERROR: No graph mode detected - please specify either fullg/traing")
        exit() 
    dataloader_test = DataLoader(
            torch.arange(g.number_of_nodes('track')),
            batch_size=32,
            collate_fn=collator.collate_test,
            num_workers=0)

    model = model.eval()
    model = model.cuda()
    device = torch.device('cuda:0')
    # ipdb.set_trace() 
    #GENERATE REPRESENTATIONS 
    all_features_cpu = [] 
    all_ids = [] 
    dataloader_it = iter(dataloader_test)
    idx = 0 
    for blocks in tqdm(dataloader_it):
        idx +=1 
        with torch.no_grad():
            for i in range(len(blocks)): 
                blocks[i] = blocks[i].to(device)
            features = model.get_repr(blocks)
            all_features_cpu.append(features.cpu().numpy()) 
            
    all_features_array = np.concatenate(all_features_cpu)
   
    #SAVE
    if output_path:
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        file_path = os.path.join(output_path,'embeddings_as_array_{}.pkl'.format(mode))
        print("***Saving Track Embeddings to {}***".format(file_path))
        pickle.dump(all_features_array, open(file_path, "wb"))
    return all_features_array



