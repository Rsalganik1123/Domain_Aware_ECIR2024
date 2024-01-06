import numpy as np
import dgl
import torch
# import ipdb
import os 
os.environ["OMP_NUM_THREADS"] = "1"

from torch.utils.data import IterableDataset, DataLoader
from .build import GRAPH_SAMPLER_REGISTRY


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.seed = 47
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                                        random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None, remove_self_loops=False ):
        blocks = []
        # ipdb.set_trace()
        for sampler in self.samplers:
            dgl.seed(self.seed)
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    # print(old_frontier)
                    # print(frontier)
                    # print(frontier.edata['weights'])
                    # frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
                # frontier = RemoveSelfLoop(frontier)
            if remove_self_loops: 
                self_loop_func = dgl.transforms.RemoveSelfLoop()
                frontier = self_loop_func(frontier)
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]
        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler #neighborhood sampler sampler 
        self.ntype = ntype
        self.g = g

    def collate_train(self, batches): 
        heads, tails, neg_tails = batches[0] #batches come from Node Sampler 
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return blocks

def assign_features_to_blocks(blocks, g, ntype='track'):
    data = blocks[0].srcdata
    for col in g.nodes[ntype].data.keys():
        # if col == dgl.NID:
        #     print("NOPE")
        #     continue
        induced_nodes = data[dgl.NID]
        data[col] = g.nodes[ntype].data[col][induced_nodes]

    data = blocks[-1].dstdata
    for col in g.nodes[ntype].data.keys():
        # if col == dgl.NID:
        #     continue
        induced_nodes = data[dgl.NID]
        data[col] = g.nodes[ntype].data[col][induced_nodes]


@GRAPH_SAMPLER_REGISTRY.register('DEFAULT')
def build_default_neighbor_sampler(g, cfg):
    sampler_cfg = cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER

    neighbor_sampler = NeighborSampler(g, cfg.DATASET.USER, cfg.DATASET.ITEM,
                                       random_walk_length=sampler_cfg.RANDOM_WALK_LENGTH,
                                       random_walk_restart_prob=sampler_cfg.RANDOM_WALK_RESTART_PROB,
                                       num_random_walks=sampler_cfg.NUM_RANDOM_WALKS,
                                       num_neighbors=sampler_cfg.NUM_NEIGHBORS,
                                       num_layers=sampler_cfg.NUM_LAYERS)
    collator = PinSAGECollator(neighbor_sampler, g, cfg.DATASET.ITEM)
    return neighbor_sampler, collator