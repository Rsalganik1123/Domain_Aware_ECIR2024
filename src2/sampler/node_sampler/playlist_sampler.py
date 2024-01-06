import torch 
import dgl 
import numpy as np 
from torch.utils.data import IterableDataset, Dataset 

class StreamingPlaylistBatchSampler(IterableDataset):
    def __init__(self, g, batch_size):
        self.g = g
        self.batch_size = batch_size
       
    def __iter__(self):
        while True:
            playlists = torch.randint(0, self.g.number_of_nodes('playlist'), (self.batch_size,))
            yield playlists 

class PlaylistBatchSampler(Dataset):
    def __init__(self, g):
        self.g = g

    def get_pos(self, playlist): 
        pos_graph, indices = dgl.khop_out_subgraph(self.g, {'playlist':playlist}, 1,  
                                  relabel_nodes=True, store_ids=True, output_device=None)
        return pos_graph 

    def get_neg(self, playlist, pos_graph): 
        neg_graph, indices = dgl.khop_out_subgraph(self.g, {'playlist':playlist}, 1,  
                                  relabel_nodes=True, store_ids=True, output_device=None)
        neg_graph.remove_edges(neg_graph.edges(etype='contains', form='eid'), 'contains')
        neg_graph.remove_edges(neg_graph.edges(etype='contained_by', form='eid'), 'contained_by')
        playlist = pos_graph.nodes['playlist'].data[dgl.NID]
        all_nodes = self.g.nodes('track')
        selected_nodes = pos_graph.nodes['track'].data[dgl.NID]
        mask = sum(all_nodes==i for i in selected_nodes).bool()
        neg_nodes = all_nodes[~mask]
        num_neg = min(len(neg_nodes), len(selected_nodes))
        src = torch.full((num_neg, ), int(playlist.int()))  
        dst = torch.tensor(np.random.choice(neg_nodes.numpy(), num_neg)) 

        neg_graph.add_edges(src, dst, etype='contains')
        neg_graph.remove_nodes(selected_nodes, ntype='track')
        return neg_graph 

    def __len__(self): 
        return self.g.num_nodes('playlist')

    def __getitem__(self, idx):
        p = self.g.nodes('playlist')[idx]
        pos = self.get_pos(p)
        neg = self.get_neg(p, pos)
        return p, pos, neg 