
import os
import torch
import time 
import numpy as np 
from src2.model.build import build_model
from src2.graph_build.data_load import build_dataset
from src2.sampler.graph_sampler.build import build_graph_sampler
from src2.sampler.node_sampler.build import build_nodes_sampler
from src2.sampler.node_sampler.playlist_sampler import PlaylistBatchSampler
from src2.model.optimizer import build_optimizer
from src2.model.layers import UsertoItemScorer, UsertoItemScorer_alone 
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import wandb 
import dgl 
from tqdm import tqdm 
import ipdb
import pickle 

def train_epoch(cfg, model, dataloader_train, dataloader_val, optimizer, tb_writer, global_steps, scaler, epoch, train_g, valid=True):
    """
    :param cfg:
    :param model:
    :param dataloader:
    :param optimizer:
    :param tb_writer:
    :param global_steps:
    :return:
    """
    b = time.time() 
    model.train() # model = model.train()
    device = torch.device('cuda:0')
    dataloader_it = iter(dataloader_train)
    loss_records, train_aucs, train_aps, train_loss = [], [], [] , [] 
    
    for batch_id in tqdm(range(cfg.TRAIN.BATCHES_PER_EPOCH)):
        
        pos_graph, neg_graph, blocks = next(dataloader_it)
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
            # print(batch_id, loss, auc)
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/batch', loss, global_steps)
                tb_writer.add_scalar('AUC/batch', auc, global_steps)
            if wandb_logging: 
                wandb.log({"loss/batch": loss, "AUC/batch": auc, "AP/batch": ap, 'epoch': epoch})
        loss_records.append([loss, auc])
        global_steps += 1
    print("EPOCH:{}, AVG TRAIN LOSS:{}, AVG TRAIN AUC:{}, AVG TRAIN AP:{}".format(epoch, np.mean(train_loss), np.mean(train_aucs), np.mean(train_aps)))
    
    valid_loss, valid_auc = [0], [0]  

    if valid:
        dataloader_val_it = iter(dataloader_val) 
        valid_loss, valid_auc, valid_ap = [], [], [] 
        model.eval()
        with torch.no_grad():
            for batch_id in range(100): 
                pos_graph, neg_graph, blocks = next(dataloader_val_it)
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
                wandb.log({"valid_loss/batch":np.mean(valid_loss), "valid_AUC/batch": np.mean(valid_auc),"valid_AP/batch": np.mean(valid_ap), 'epoch': epoch})
        a = time.time() 
        print("VALIDATION PERFORMANCE: LOSS:{}, AUC:{}, AP:{} TIME PER EPOCH:{} sec".format(np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_ap), (a-b))) 
    
    return loss_records, [np.mean(valid_loss), np.mean(valid_auc)], global_steps


def train(cfg):
    output_path = cfg.OUTPUT_PATH
    print("OUTPUT_PATH", output_path, "NAME:", output_path.split('/')[-2])
    assert output_path != ''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    tb_path = os.path.join(output_path, 'tb')
    tb_writer = SummaryWriter(tb_path)
    
    if wandb_logging: 
        wandb.config = cfg 
        wandb.init(
            project='MusicSAGE train', 
            name=output_path.split('/')[-2],
            notes='name:{}, feature set:{}, loss:{}'.format(output_path.split('/')[-2], cfg.MODEL.PINSAGE.PROJECTION.FEATURES, cfg.TRAIN.LOSS), 
            tags=['feature_tests', 'pinsage'], 
            config = cfg 
        )
        wandb.run.name = output_path.split('/')[-2]
        cfg.WANDB = wandb.run.name
    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))
    
    
    print("Saving Checkpoints to:{}".format(checkpoints_path))

    # load data
    g, train_g, val_g = build_dataset(cfg)
    print("Loaded graphs: Full:{}, \nTrain:{}, \nValidation:{}".format(g, train_g, val_g))
    
    # load model
    model = build_model(g, cfg)

    # load samplers - train
    train_nodes_sampler = build_nodes_sampler(train_g, cfg)
    neighbor_sampler, collator = build_graph_sampler(train_g, cfg)
    dataloader_train = DataLoader(train_nodes_sampler, collate_fn=collator.collate_train, num_workers=2)

    # load samplers - val 
    val_nodes_sampler = build_nodes_sampler(val_g, cfg)
    neighbor_sampler, collator = build_graph_sampler(val_g, cfg)
    dataloader_val = DataLoader(val_nodes_sampler, collate_fn=collator.collate_train, num_workers=2) #collate train or test?

    # scaler
    if cfg.FP16:
        scaler = GradScaler()
    else:
        scaler = None

    # model to gpu
    model = model.cuda()

    # build optimizer
    global_steps = 0
    optimizer = build_optimizer(model, cfg)
    lr_steps = cfg.TRAIN.SOLVER.STEP_LRS
    lr_steps = {x[0]: x[1] for x in lr_steps}
    base_lr = cfg.TRAIN.SOLVER.BASE_LR
    all_epochs = cfg.TRAIN.EPOCHS
    best_val_auc = 0 
    for epoch_idx in range(all_epochs):
        if cfg.TRAIN.SOLVER.DECAY: 
            lr = lr_steps.get(epoch_idx, base_lr)
            for param in optimizer.param_groups:
                param['lr'] = lr
        else: 
            lr = base_lr
        print('Start epoch {0}, lr {1}'.format(epoch_idx, lr))
        # update learning rate
        
        loss_records, valid_loss_records, global_steps = train_epoch(cfg, model, dataloader_train, dataloader_val, optimizer, tb_writer, global_steps, scaler, epoch_idx, train_g, valid=True)
        print("GRAPH", train_g, "GRAPH_DATA",  train_g.nodes['track'].data.keys())
        
        
        if valid_loss_records[1] > best_val_auc: 
            best_val_auc = valid_loss_records[1]

        if cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER.ADAPTIVE and cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER != 'DEFAULT': 
            train_nodes_sampler.incr_epoch() 
            print("incremented sampler: ", train_nodes_sampler.epoch)
        save_state = {
            'global_steps': global_steps,
            "epoch": epoch_idx + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            'loss_records': loss_records, 
            'valid_loss_records': valid_loss_records
        }
        backup_fpath = os.path.join(checkpoints_path, "model_bak_%06d.pt" % (epoch_idx,))
        print("saving checkpoint for epoch:{} to:{}".format(epoch_idx, backup_fpath))
        torch.save(save_state, backup_fpath)
    return 1 - best_val_auc

    
wandb_logging = False