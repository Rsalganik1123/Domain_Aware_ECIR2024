import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
from .build import NODE_SAMPLER_REGISTRY
import numpy as np 
import pickle 



@NODE_SAMPLER_REGISTRY.register('CURRICULUM')
class CurriculumItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, cfg):
        self.g = g
        self.user_type = cfg.DATASET.USER
        self.item_type = cfg.DATASET.ITEM
        self.user_to_item_etype = list(g.metagraph()[self.user_type][self.item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[self.item_type][self.user_type])[0]
        self.batch_size = cfg.DATASET.SAMPLER.NODES_SAMPLER.BATCH_SIZE
        self.epoch = 0 
        self.k_hop = cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER.HOPS_AWAY 
        self.adaptive = cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER.ADAPTIVE
    
    def incr_epoch(self): 
        self.epoch += 1
    
    def get_tails(self, trace): 
        heads = trace[:, 0]
        mask = torch.full(heads.shape, -1)
        tail_opt = trace[:, 2]
        tails = torch.where(tail_opt != heads, tail_opt, mask)
        return tails 

    def get_neg_tails(self, trace): 
        heads = trace[:, 0]
        neg_tail_opt = trace[:, 2*self.k_hop]
        if self.adaptive: 
            idx = torch.randint(0, len(heads), (self.epoch, ))
        else: 
            idx = torch.arange(0, len(heads))
        for i in idx: 
            seed, neg = heads[i], neg_tail_opt[i]
            if neg == -1: continue 
            parent = self.g.predecessors(seed, etype='contains')
            if (self.g.has_edges_between(neg, parent, 'contained_by')).any(): 
                neg_tail_opt[i] = -1 
        return neg_tail_opt
    
    def fillna(self, opt): 
        random = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
        return torch.where(opt != -1, opt, random)
    
    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            trace, eids, types = dgl.sampling.random_walk(
                self.g,
                heads,
                return_eids=True,
                metapath=['contained_by', 'contains']*self.k_hop)
            tails = self.get_tails(trace)
            neg_tails = self.fillna(self.get_neg_tails(trace)) 
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]




@NODE_SAMPLER_REGISTRY.register('PAGE_RANK')
class PGItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, cfg):
        self.g = g
        self.user_type = cfg.DATASET.USER
        self.item_type = cfg.DATASET.ITEM
        self.user_to_item_etype = list(g.metagraph()[self.user_type][self.item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[self.item_type][self.user_type])[0]
        self.batch_size = cfg.DATASET.SAMPLER.NODES_SAMPLER.BATCH_SIZE
        self.candidates = pickle.load(open('/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/small_100_10/pg_candidates.pkl', "rb"))
    
    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = 0 
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

@NODE_SAMPLER_REGISTRY.register('DEFAULT')
class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, cfg, device='cpu'):
        self.g = g
        self.user_type = cfg.DATASET.USER
        self.item_type = cfg.DATASET.ITEM
        self.user_to_item_etype = list(g.metagraph()[self.user_type][self.item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[self.item_type][self.user_type])[0]
        self.batch_size = cfg.DATASET.SAMPLER.NODES_SAMPLER.BATCH_SIZE
        self.seed = cfg.seed 
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
    
    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,), generator=self.generator)
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,), generator=self.generator)
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]