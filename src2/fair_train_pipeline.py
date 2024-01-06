import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import dgl 
import os
import torch
import numpy as np 
import time 
from tqdm import tqdm 
import random
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time 
import wandb 
import pickle 
from src2.model.build import build_model
from src2.graph_build.data_load import build_dataset
from src2.sampler.graph_sampler.build import build_graph_sampler
from src2.sampler.node_sampler.build import build_nodes_sampler
from src2.model.optimizer import build_optimizer
from src2.fairness.fairness_utils import *
from src2.model.early_stopping import * 
from src2.utils.misc import * 
from src2.utils.save_res import * 
from src2.eval.validation_eval import rec_validation 

def validation_tests(cfg, checkpoint_path):
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']
    val_data = df_interactions.loc[val_indices]

    g, train_g, val_g = build_dataset(cfg)
    model = build_model(g, cfg)
    model_state = torch.load(checkpoint_path, map_location='cpu')
    print("***Loaded Checkpoint from:{}***".format(checkpoint_path))
    model.load_state_dict(model_state['model_state'])
    
    neighbor_sampler, collator = build_graph_sampler(g, cfg) 
    
    dataloader_test = DataLoader(
            g.nodes('track'),
            batch_size=32,
            collate_fn=collator.collate_test,
            num_workers=0)

    avg_recall = rec_validation(cfg, model, dataloader_test, val_data, df_items)
    print(avg_recall)
    return 0 


def fair_train(cfg, model, dataloader_train, dataloader_val, optimizer, global_steps, scaler, epoch, logging, tb_writer=None, valid=False): 
    wandb_logging = logging
    model.train() 
    device = torch.device('cuda:0')
    
    # fairness_t_losses, fairness_v_losses, fairness_perfs = [], [], []
    loss_records, train_aucs, train_aps, train_loss, train_fairness = [], [], [] , [] , []   
    # ipdb.set_trace()
    for batch_id in tqdm(range(cfg.TRAIN.BATCHES_PER_FAIRNESS_EPOCH)):
        pos_graph, neg_graph, blocks = next(iter(dataloader_train)) 
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        pos_score, neg_score, utility_loss, auc, ap, h_item = model(pos_graph, neg_graph, blocks)
        optimizer.zero_grad()
        if cfg.FAIR.METHOD == 'REDRESS': 
            utility_loss.backward(retain_graph=True)
            fair_performance, pred_sim, lambdas  = calc_fair_loss(cfg, model, blocks, sim_feat=cfg.FAIR.FEAT_SET)
            pred_sim.backward(cfg.FAIR.FAIRNESS_BALANCE*lambdas)
            optimizer.step() 
        elif cfg.FAIR.METHOD == 'SCORE_REG':
            pos_score = pos_score.cuda() 
            neg_score = neg_score.cuda() 
            utility_loss = - torch.sum((pos_score - neg_score).sigmoid().log())                
            fair_performance =  - torch.sum((1 -(pos_score[0] + neg_score[0]).abs() .tanh() ).log()) 
            print(utility_loss, fair_performance, pos_score[0], neg_score[0])
            loss = utility_loss*(1-cfg.FAIR.FAIRNESS_BALANCE) + fair_performance*cfg.FAIR.FAIRNESS_BALANCE
            loss.backward()
        else: 
            print('ERROR: NO SAMPLING METHOD SPECIFID')
            exit() 
        utility_loss = utility_loss.item()
        train_loss.append(utility_loss)
        train_aucs.append(auc)
        train_aps.append(ap)
        train_fairness.append(fair_performance)
        if batch_id % 100 == 0:
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/batch', loss, global_steps)
                tb_writer.add_scalar('AUC/batch', auc, global_steps)
            if wandb_logging: 
                wandb.log({"loss/batch": utility_loss, "AUC/batch": auc, "AP/batch": ap, "FAIRNESS/batch": fair_performance, 'epoch': epoch})
        
        global_steps += 1
    print("EPOCH:{}, TRAIN UTILITY LOSS:{}, TRAIN UTILITY AUC:{}, TRAIN UTILITY AP:{}, TRAIN FAIRNESS:{} ".format(epoch, np.mean(train_loss), np.mean(train_aucs), np.mean(train_aps), np.mean(train_fairness)))
        
    valid_loss, valid_auc, valid_fairness = [0], [0] , [0]

    if valid:
        dataloader_val_it = iter(dataloader_val) 
        valid_loss, valid_auc, valid_ap, valid_fairness = [], [], [] , [] 
        model.eval() #model = model.eval() 
        with torch.no_grad():
            for batch_id in range(10):
                pos_graph, neg_graph, blocks = next(iter(dataloader_val)) 
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                pos_score, neg_score, val_loss, val_auc, val_ap, h_item  = model(pos_graph, neg_graph, blocks)
                fair_performance, pred_sim, lambdas  = calc_fair_loss(cfg, model, blocks, sim_feat=cfg.FAIR.FEAT_SET)

                valid_loss.append(val_loss.item())
                valid_auc.append(val_auc.item())
                valid_ap.append(val_ap.item())
                valid_fairness.append(fair_performance)
            if wandb_logging: 
                    wandb.log({"valid_loss/batch":np.mean(valid_loss), "valid_AUC/batch": np.mean(valid_auc),"valid_AP/batch": np.mean(valid_ap),  "FAIRNESS/batch": np.mean(valid_fairness), 'epoch': epoch})
        print("VALIDATION PERFORMANCE: UTILITY LOSS:{}, UTILITY AUC:{}, UTILITY AP:{} FAIRNESS:{} sec".format(np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_ap), np.mean(valid_fairness))) 
    return [np.mean(train_loss), np.mean(train_aucs), np.mean(train_fairness)], [np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_fairness)], np.mean(valid_fairness),  global_steps


def utility_train(cfg, model, dataloader_train, dataloader_val, optimizer, global_steps, scaler, epoch, logging, tb_writer=None, valid=False):
    wandb_logging = logging 
    b = time.time() 
    model.train() # model = model.train()
    device = torch.device('cuda:0')
    dataloader_it = iter(dataloader_train)
    loss_records, train_aucs, train_aps, train_loss = [], [], [] , [] 
    
    for batch_id in tqdm(range(cfg.TRAIN.BATCHES_PER_UTILITY_EPOCH)):   
        pos_graph, neg_graph, blocks = next(iter(dataloader_train)) 
        # Copy to GPU
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        #Train
        pos_score, neg_score, loss, auc, ap, h_item = model(pos_graph, neg_graph, blocks)
        #Back Prop 
        optimizer.zero_grad()
        loss.backward()
        if cfg.TRAIN.SOLVER.GRAD_CLIPPING: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        train_aucs.append(auc)
        train_aps.append(ap)
        if batch_id % 100 == 0:
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/batch', loss, global_steps)
                tb_writer.add_scalar('AUC/batch', auc, global_steps)
            if wandb_logging: 
                wandb.log({"loss/batch": loss, "AUC/batch": auc, "AP/batch": ap, "FAIRNESS/batch": 0.0, 'epoch': epoch})
        loss_records.append([loss, auc])
        global_steps += 1
    print("EPOCH:{}, AVG TRAIN LOSS:{}, AVG TRAIN AUC:{}, AVG TRAIN AP:{}".format(epoch, np.mean(train_loss), np.mean(train_aucs), np.mean(train_aps)))

    valid_loss, valid_auc = [0], [0]  

    if valid:
        dataloader_val_it = iter(dataloader_val) 
        valid_loss, valid_auc, valid_ap = [], [], [] 
        model.eval() #model = model.eval() 
        with torch.no_grad():
            for batch_id in range(100): 
                pos_graph, neg_graph, blocks = next(iter(dataloader_val)) 
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                pos_score, neg_score, val_loss, val_auc, val_ap, h_item = model(pos_graph, neg_graph, blocks)
                valid_loss.append(val_loss.item())
                valid_auc.append(val_auc.item())
                valid_ap.append(val_ap.item())
            if tb_writer is not None:
                    tb_writer.add_scalar('valid_loss/avg', np.mean(valid_loss), global_steps)
                    tb_writer.add_scalar('valid_auc/avg', np.mean(valid_auc), global_steps)
            if wandb_logging: 
                wandb.log({"valid_loss/batch":np.mean(valid_loss), "valid_AUC/batch": np.mean(valid_auc),"valid_AP/batch": np.mean(valid_ap),  "FAIRNESS/batch": 0.0, 'epoch': epoch})
        a = time.time() 
        print("VALIDATION PERFORMANCE: LOSS:{}, AUC:{}, AP:{} TIME PER EPOCH:{} sec".format(np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_ap), (a-b))) 
    
    # return np.mean(valid_loss), global_steps
    return [np.mean(train_loss),np.mean(train_aucs)], [np.mean(valid_loss), np.mean(valid_auc)], global_steps

    
def fair_train_main(cfg, pretrained_path=None): 
   
    dgl.seed(47)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)   
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb_logging = False 

    output_path = cfg.OUTPUT_PATH
    
    assert output_path != ''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    best_epoch_path = os.path.join(output_path, 'best_epoch')
    if not os.path.exists(best_epoch_path):
        os.mkdir(best_epoch_path)

    # load data
    print("***loading data***")
    g, train_g, val_g = build_dataset(cfg)

    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']
    val_data = df_interactions.loc[val_indices]

    init_fairness, utility_fairness, fair_fairness = 0, 0, 0 

     # load samplers - train
    nodes_sampler = build_nodes_sampler(train_g, cfg)
    neighbor_sampler, collator = build_graph_sampler(train_g, cfg)
    dataloader_train = DataLoader(nodes_sampler, collate_fn=collator.collate_train, num_workers=0) #2


    # load samplers - val 
    nodes_sampler = build_nodes_sampler(val_g, cfg)
    neighbor_sampler, collator = build_graph_sampler(val_g, cfg)
    dataloader_val = DataLoader(nodes_sampler, collate_fn=collator.collate_train, num_workers=0) #2

    #load sampler - entire graph 
    neighbor_sampler, collator = build_graph_sampler(g, cfg)
    dataloader = DataLoader(
            torch.arange(g.number_of_nodes('track')),
            batch_size=32,
            collate_fn=collator.collate_test,
            num_workers=0) #2
            
    # load model
    print("***loading model and dataloaders***")
    model = build_model(g, cfg)
    optimizer = build_optimizer(model, cfg)
    try: 
        if wandb_logging: 
            wandb.config = cfg 
            wandb.init(
                project='MusicSAGE train', 
                name=output_path.split('/')[-2],
                notes='name:{}, feature set:{}, loss:{}'.format(output_path.split('/')[-2], cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES, cfg.TRAIN.LOSS), 
                tags=['fairness_tests', 'pinsage'], 
                config = cfg 
            )
            wandb.run.name = output_path.split('/')[-2]
            cfg.WANDB = wandb.run.name
    except Exception as e: 
        print("logging failed", e)
        wandb_logging = False 
    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))
    
    epoch = 0 
    
    if pretrained_path: 
        model, optimizer, epoch, global_steps = load_checkpoints(model, optimizer, pretrained_path)
    
        optimizer_to(optimizer,'cuda')
        cfg.TRAIN.UTILITY_EPOCHS = cfg.TRAIN.UTILITY_EPOCHS - epoch if cfg.TRAIN.UTILITY_EPOCHS - epoch > 0 else 0 
     
    # scaler
    if cfg.FP16:
        scaler = GradScaler()
    else:
        scaler = None

    # model to gpu
    model = model.cuda()
    if cfg.RUN.EARLY_STOPPING: 
        early_stopper = EarlyStopper(tolerance = 5, delta = 0.01) 
    # build optimizer
    global_steps = 0
    lr_steps = cfg.TRAIN.SOLVER.STEP_LRS
    lr_steps = {x[0]: x[1] for x in lr_steps}
    base_lr = cfg.TRAIN.SOLVER.BASE_LR

    if cfg.RUN.GLOBAL_FAIR_CALC: 
        init_fairness = calc_fair_loss_entireg(cfg, model, dataloader, g, feat=cfg.FAIR.FEAT_SET)
        print("init fairness", init_fairness)

    utility_train_loss,  utility_train_auc, utility_val_loss, utility_val_auc=  -1.0, -1.0, -1.0, -1.0
    fair_train_loss, fair_train_auc, fair_val_loss, fair_val_auc = -1.0, -1.0, -1.0, -1.0

    print("***Utility mode: {} epochs***".format(cfg.TRAIN.UTILITY_EPOCHS))

    best_recall = 0 
    for epoch_idx in range(cfg.TRAIN.UTILITY_EPOCHS):
        if cfg.TRAIN.SOLVER.DECAY: 
            lr = lr_steps.get(epoch_idx, base_lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
        else: lr = base_lr
        avg_train_performance, avg_valid_performance, global_steps  = utility_train(cfg, model, dataloader_train, dataloader_val, optimizer, global_steps, scaler, epoch_idx, wandb_logging, valid=False)
        utility_train_loss, utility_train_auc = avg_train_performance
        utility_val_loss, utility_val_auc = avg_valid_performance
        
        avg_recall = rec_validation(cfg, model, dataloader, val_data, df_items, mode = 'utility')

        print("EPOCH:{}, BEST_RECALL@20:{}, CURRENT_RECALL@20:{}".format(epoch_idx, best_recall, avg_recall))
        
        save_state = {
            'global_steps': global_steps,
            "epoch": epoch_idx + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            'loss_records': avg_train_performance, 
            'mode': 'utility',
            'valid_loss_records': avg_valid_performance
        }

        if best_recall < avg_recall: 
            best_recall = avg_recall 
            backup_fpath = os.path.join(best_epoch_path, "u_model_bak_%06d.pt" % (epoch_idx,))
            torch.save(save_state, backup_fpath)

        backup_fpath = os.path.join(checkpoints_path, "u_model_bak_%06d.pt" % (epoch_idx,))
        if not cfg.RUN.TESTING: 
            print("saving checkpoint for epoch:{} to:{}".format(epoch_idx, backup_fpath))
            torch.save(save_state, backup_fpath)
        if cfg.RUN.EARLY_STOPPING: 
            if early_stopper(utility_val_loss, global_steps, epoch_idx, model, optimizer): 
                print("Early Stopping at EPOCH:{}".format(epoch_idx))
                break 
        
    if cfg.RUN.GLOBAL_FAIR_CALC: 
        utility_fairness = calc_fair_loss_entireg(cfg, model, dataloader, g, feat = cfg.FAIR.FEAT_SET)
        print("utility fairness", utility_fairness)
    
    if cfg.RUN.EARLY_STOPPING: 
        early_stopper.__reset__() 

    if cfg.RUN.PRETRAINED:
        cfg.TRAIN.UTILITY_EPOCHS += epoch
    
    best_LT = 0
    print("***Fairness mode: {} epochs***".format(cfg.TRAIN.FAIR_EPOCHS))
    for epoch_idx in range(cfg.TRAIN.FAIR_EPOCHS):
        if cfg.TRAIN.SOLVER.DECAY: 
            lr = lr_steps.get(epoch_idx, base_lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
        else: lr = base_lr
        avg_train_performance, avg_valid_performance, avg_fairness, global_steps = fair_train(cfg, model, dataloader_train, dataloader_val, optimizer, global_steps, scaler, epoch_idx, wandb_logging, valid=False)
        fair_train_loss, fair_train_auc, fair_train_fairness = avg_train_performance
        fair_val_loss, fair_val_auc, fair_val_fairness = avg_valid_performance
        
        avg_LT = rec_validation(cfg, model, dataloader, val_data, df_items, mode = 'fairness')

        print("EPOCH:{}, BEST_LT:{}, CURRENT_LT:{}".format(epoch_idx, best_LT, avg_LT))
        
        

        save_state = {
            'global_steps': global_steps,
            "epoch": cfg.TRAIN.UTILITY_EPOCHS + epoch_idx + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            'loss_records': avg_train_performance, 
            'mode': 'fair',
            'valid_loss_records': avg_valid_performance
        }
        if best_LT < avg_LT: 
            best_LT = avg_LT
            backup_fpath = os.path.join(best_epoch_path, "u+f_model_bak_%06d.pt" % (cfg.TRAIN.UTILITY_EPOCHS + epoch_idx,))
            torch.save(save_state, backup_fpath)

        backup_fpath = os.path.join(checkpoints_path, "u+f_model_bak_%06d.pt" % (cfg.TRAIN.UTILITY_EPOCHS + epoch_idx,))
        if not cfg.RUN.TESTING: 
            print("saving checkpoint for epoch:{} to:{}".format(cfg.TRAIN.UTILITY_EPOCHS + epoch_idx, backup_fpath))
            torch.save(save_state, backup_fpath)

        
    if cfg.RUN.GLOBAL_FAIR_CALC: 
        fair_fairness = calc_fair_loss_entireg(cfg, model, dataloader, g, feat = cfg.FAIR.FEAT_SET)
        print("fair fairness", fair_fairness)

    summary = {
        'utility_train_loss': utility_train_loss, 
        'utility_train_auc': utility_train_auc, 
        'utility_valid_loss': utility_val_loss, 
        'utility_valid_auc': utility_val_auc, 
        'fair_train_loss': fair_train_loss, 
        'fair_train_auc': fair_train_auc, 
        'fair_valid_loss': fair_val_loss, 
        'fair_valid_auc': fair_val_auc,  
        'init_fairness': init_fairness, 
        'utility_fairness': utility_fairness, 
        'fair_fairness':fair_fairness
    }
    if wandb_logging: 
        save_run_performance(output_path, summary)
     

wandb_logging = False