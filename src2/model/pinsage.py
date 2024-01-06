from torch import nn
from tqdm import tqdm 
import torch 
import dgl 
# from src2.graph_build.data_load import DATASET_REGISTRY
from src2.model.layers import LinearProjector, SAGENet, ItemToItemScorer, UsertoItemScorer
from src2.model.build import MODEL_REGISTRY
from src2.model.loss_fn import LOSS_REGISTRY, compute_auc, compute_ap
from torch.utils.data import IterableDataset, DataLoader



class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, cfg):
        super().__init__()
        self.hidden_size = cfg.MODEL.PINSAGE.HIDDEN_SIZE
        self.proj = LinearProjector(full_graph, cfg)
        self.sage = SAGENet(self.hidden_size, cfg.MODEL.PINSAGE.LAYERS, cfg.MODEL.PINSAGE.DROPOUT)
        self.scorer = ItemToItemScorer(full_graph, cfg)
        # self.second_scorer = UsertoItemScorer(full_graph, cfg)
        self.loss_fn = LOSS_REGISTRY[cfg.TRAIN.LOSS]
        self.IPW = cfg.FAIR.IPW 
        self.POP_FEAT = 'appear_raw' 
        if cfg.MODEL.PINSAGE.REPRESENTATION_NORMALIZE:
            self.norm = nn.LayerNorm(cfg.MODEL.PINSAGE.HIDDEN_SIZE)
        else:
            self.norm = None
        
    def forward(self, pos_graph, neg_graph, blocks):
        
        h_item = self.get_repr(blocks)
        
        if self.IPW: 
            pop = self.get_pop(blocks)
            # print("POP shape", pop.shape)
            # print(blocks)
            pos_score = self.scorer(pos_graph, h_item, pop=pop)
            # print("POS_SCORE shape", pos_score.shape)
            neg_score = self.scorer(neg_graph, h_item, pop=pop)
            # print("NEG_SCORE shape", neg_score.shape)

        else:     
            pos_score = self.scorer(pos_graph, h_item)
            # print("POS_SCORE shape", pos_score.shape)
            neg_score = self.scorer(neg_graph, h_item)
            # print("NEG_SCORE shape", neg_score.shape)
            
        # exit()
        loss = self.loss_fn(pos_score, neg_score)
        auc = compute_auc(pos_score, neg_score)
        ap = compute_ap(pos_score, neg_score) 
        # print(pos_score, neg_score, loss)
        # edge_scores = self.second_scorer(blocks[-1], h_item)
        # print(pos_graph.ndata[dgl.NID], blocks[-1].dstnodes('track'), h_item.shape)
        # exit() 
        return pos_score, neg_score, loss, auc, ap, h_item

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        # h_item = blocks[0].srcdata['feats']
        # h_item_dst = blocks[-1].dstdata['feats']
        h = h_item_dst + self.sage(blocks, h_item)
        if self.norm:
            h = self.norm(h)
        return h
    
    def get_pop(self, blocks): 
        pop = blocks[-1].dstdata[self.POP_FEAT].reshape(-1, 1)
        return pop 

    def inference(self, dataloader, cfg, feature_set=None): 
        if not feature_set: 
            feature_set = self.proj.all_features
        embeddings, features = [], [] 
        for blocks in tqdm(iter(dataloader)):
            with torch.no_grad():
                for i in range(len(blocks)): 
                    blocks[i] = blocks[i].to('cuda')
                emb = self.get_repr(blocks)
                embeddings.append(emb.cpu().numpy()) 
        embeddings = np.concatenate(embeddings)
        return torch.tensor(embeddings)  
    
@MODEL_REGISTRY.register('PINSAGE')
def build_pinsage_model(full_graph, cfg):
    model = PinSAGEModel(full_graph, cfg)
    return model