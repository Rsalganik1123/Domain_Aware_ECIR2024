import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import numpy as np 
# import ipdb


def init_embeddings(g, cfg):
    emb_types = cfg.MODEL.PINSAGE.PROJECTION.EMB #dictionary of embedding sizes 
    data = g.nodes[cfg.DATASET.ITEM].data #all the available data 
    module_dict = torch.nn.ModuleDict()

    for key, size in emb_types:
        if key in cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES: 
            module_dict[key] = torch.nn.Embedding(data[key].max() + 1, size)

    return module_dict



class LinearProjector(torch.nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, cfg):
        super().__init__()
        
        self.ntype = cfg.DATASET.ITEM #set which node type gets embeddings --> track
        self.embeddings = init_embeddings(full_graph, cfg) #generate embedding space for all the tracks in the graph
        self.hidden_size = cfg.MODEL.PINSAGE.HIDDEN_SIZE #define linear layer size 
        self.concat_feature_types = cfg.MODEL.PINSAGE.PROJECTION.CONCAT #list of scalar valued features to use 
        self.all_features = cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES #list of features to use 

        
        self.album_features = [x for x in self.all_features if x in ['img_emb']] #array of all img embeddings
        self.text_features = [x for x in self.all_features if x in ['track_name_emb']] #array of all BERT embeddings
        # self.genre_features = [x for x in self.all_features if x in ['genres']] #array of all genre embeddings

        data = full_graph.nodes[cfg.DATASET.ITEM].data #getting all the features from the graph

        if len(self.album_features) > 0: #if we are using img feat
            album_feature_size = 0
            for key in self.album_features: #figure out what size it should be
                _, dim = data[key].shape
                album_feature_size += dim  
            self.fc_album = torch.nn.Linear(album_feature_size, self.hidden_size) #instantiate a linear layer of that size
        else:
            self.fc_album = None
        
        if len(self.text_features) > 0: #if we are using BERT feat
            text_feature_size = 0
            for key in self.text_features: #figure out what size it should be
                _, dim = data[key].shape
                text_feature_size += dim
            self.fc_text = torch.nn.Linear(text_feature_size, self.hidden_size) #instantiate a linear layer of that size
        else:
            self.fc_text = None
    
        # if len(self.genre_features) > 0: #if we are using genre_vector
        #     genre_feature_size = 0
        #     for key in self.genre_features: #figure out what size it should be
        #         _, dim = data[key].shape
        #         genre_feature_size += dim
        #     self.fc_genre = torch.nn.Linear(genre_feature_size, self.hidden_size) #instantiate a linear layer of that size
        # else:
        #     self.fc_genre = None

        concat_size = 0 #
        for key in self.concat_feature_types:

            if key in self.embeddings:
                embs = self.embeddings[key]
                concat_size += embs.embedding_dim
            else:
                _, dim = data[key].shape
                concat_size += dim

        if self.fc_album is not None:
            concat_size += self.hidden_size
        
        if self.fc_text is not None:
            concat_size += self.hidden_size

        self.concat_size = concat_size
        if concat_size > 0:
            self.fc = torch.nn.Linear(concat_size, self.hidden_size)
        else:
            self.fc = None
        self.add_feature_types = cfg.MODEL.PINSAGE.PROJECTION.ADD
        if cfg.MODEL.PINSAGE.PROJECTION.NORMALIZE:
            self.norm = torch.nn.LayerNorm(self.hidden_size)
        else:
            self.norm = None
    
    def forward(self, ndata):

        features = {}
        for key in self.all_features:

            if key in self.embeddings:
                module = self.embeddings[key]
                value = module(ndata[key])
            else:
                value = ndata[key]
            features[key] = value

        projection = 0
        for key in self.add_feature_types:
            projection = projection + features[key]

        if len(self.album_features) > 0:
            album_feature = torch.cat([features[x] for x in self.album_features], dim=1)
            album_feature = self.fc_album(album_feature)
        else:
            album_feature = None
            
        if len(self.text_features) > 0:
            text_feature = torch.cat([features[x] for x in self.text_features], dim=1)
            text_feature = self.fc_text(text_feature)
        else:
            text_feature = None

        # if len(self.genre_features) > 0:
        #     genre_feature = torch.cat([features[x] for x in self.genre_features], dim=1)
        #     genre_feature = self.fc_genre(genre_feature)
        # else:
        #     genre_feature = None

        concat_features = []
        for key in self.concat_feature_types:
            concat_features.append(features[key])
        if album_feature is not None:
            concat_features.append(album_feature)
        if text_feature is not None:
            concat_features.append(text_feature)
        # if genre_feature is not None:
        #     concat_features.append(genre_feature)
        if len(concat_features) > 0:
            concat_features = torch.cat(concat_features, dim=1)
            projection = projection + self.fc(concat_features)
        if self.norm:
            projection = self.norm(projection)

        return projection

class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropout, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        # print("src shape", h_src.shape, 'dst shape', h_dst.shape)
        # h_src = g.srcdata['feats']
        # h_dst = g.dstdata['feats']
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z

class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers, dropout):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims, dropout))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        blocks[-1].dstdata['feats'] = h
        return h

class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, cfg):
        super().__init__()

        if cfg.MODEL.PINSAGE.SCORER_BIAS:
            n_nodes = full_graph.number_of_nodes(cfg.DATASET.USER)
            self.bias = nn.Parameter(torch.zeros(n_nodes))
        else:
            self.bias = None

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h, pop=None):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            if pop != None: 
                item_item_graph.ndata['ipw'] = 1/pop 
                item_item_graph.apply_edges(fn.u_dot_v('ipw', 'ipw', 'ipw_score'))
            if self.bias:
                item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s'][:, 0]
            if pop != None: 
                ipw_pair_score = item_item_graph.edata['ipw_score'][:, 0]
                pair_score = pair_score * ipw_pair_score
        return pair_score


class UsertoItemScorer_alone(nn.Module):
    def __init__(self, full_graph,  cfg):
        # self.ITEM_USER_EDGE = cfg.DATASET.ITEM_USER_EDGE
        self.g = full_graph
        
        super().__init__()
    
    def forward(self, p_nodes = None):
        """
        full_graph : graph consists of edges connecting the playlists and tracks
        """
        if len(p_nodes) == 0: 
            p_nodes = self.g.nodes('playlist')
        with self.g.local_scope():
            print("ITEM TO USER SCORER") 
            self.g.nodes['track'].data['h'] = self.g.nodes['track'].data['emb']
            self.g.nodes['playlist'].data['h'] = torch.zeros((self.g.num_nodes('playlist'), self.g.nodes['track'].data['emb'].shape[1]))
            # print("playlist features before", self.g.nodes['playlist'].data['h'])
            for p_node in p_nodes: 
                u, v, track_edges = self.g.out_edges(p_node, etype='contains', form='all')
                # print("playlist:{}, u:{}, v:{}, track_edges:{}".format(p_node, u, v,track_edges)) 
                gen_edges = np.random.choice(track_edges, 2) 
                # print("gen_edges", gen_edges)
                self.g['contained_by'].prop_edges([gen_edges], fn.copy_src('h', 'm'),
                                fn.mean('m', 'h'), etype='contained_by')
                # print("Updated playlist features", self.g.nodes['playlist'].data['h'])
            self.g.apply_edges(fn.u_dot_v('h', 'h', 's'))
            pair_score = self.g.edata['s']
            # print(pair_score[('playlist', 'contains', 'track')])
        return pair_score[('playlist', 'contains', 'track')]

class UsertoItemScorer(nn.Module):
    def __init__(self):
        # self.ITEM_USER_EDGE = cfg.DATASET.ITEM_USER_EDGE
        # self.g = full_graph
        super().__init__()
    def get_playlist_reps(self, g, h, k=2): 
        with g.local_scope():
            print("ITEM TO USER SCORER") 
            g.nodes['track'].data['h'] = h
            g.nodes['playlist'].data['h'] = torch.zeros((g.num_nodes('playlist'), h.shape[1]))
            for p_node in g.nodes('playlist'): 
                u, v, track_edges = g.out_edges(p_node, etype='contains', form='all')
                # print("playlist:{}, u:{}, v:{}, track_edges:{}".format(p_node, u, v,track_edges)) 
                gen_edges = np.random.choice(track_edges, k) 
                # print("gen_edges", gen_edges)
                g['contained_by'].prop_edges([gen_edges], fn.copy_src('h', 'm'),
                                fn.mean('m', 'h'), etype='contained_by')
            return g.nodes['playlist'].data['h']
    
    def forward(self, sub_g):
        with sub_g.local_scope():
            sub_g.apply_edges(fn.u_dot_v('h', 'h', 's'))
            pair_score = sub_g.edata['s']
            return pair_score[('playlist', 'contains', 'track')] 
                    

    
class UI_Embd(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.W = nn.Linear(input_size, input_size)
        self.reset_parameters()
        

    def reset_parameters(self):
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        # print("src shape", h_src.shape, 'dst shape', h_dst.shape)
        # h_src = g.srcdata['feats']
        # h_dst = g.dstdata['feats']
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z
    