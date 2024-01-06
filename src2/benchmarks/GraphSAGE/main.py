from src2.benchmarks.GraphSAGE.graph_build import * 
from src2.benchmarks.GraphSAGE.model import * 
from tqdm import tqdm 
from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, negative_sampler
import os 
import pandas as pd

dataset_path = 'FILL IN'

def train(p): 
       
        if not os.path.exists(p):
                os.mkdir(p)
                os.mkdir(os.path.join(p, 'checkpoints'))
                os.mkdir(os.path.join(p, 'embeddings'))
                os.mkdir(os.path.join(p, 'recs'))

        #Build graph 
        g, reverse_eids, train_eids, val_eids = build()

        #Create samplers 

        device = 'cuda'
        g = g.to(device)
        train_eids = train_eids.to(device) 
        val_eids = val_eids.to(device)

        sampler = NeighborSampler([10, 25], prefetch_node_feats=['feat'])

        sampler = as_edge_prediction_sampler(
                sampler,
                negative_sampler=negative_sampler.Uniform(5))

        train_dataloader = DataLoader(
                g, train_eids, sampler,
                device=device, batch_size=12, shuffle=True,
                drop_last=False, num_workers=0)

        val_dataloader = DataLoader(
                g, val_eids, sampler,
                device=device, batch_size=12, shuffle=True,
                drop_last=False, num_workers=0)

        #Build model/optimizer 
        in_size = g.ndata['feat'].shape[1]
        model = SAGE(in_size, 2560).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.00001)
        
        for epoch in range(10): 
                model.train()
                batch_losses =  []  #train_auc, train_ap = [], []
                for input_nodes, pair_graph, neg_pair_graph, blocks in tqdm(train_dataloader):
                        x = blocks[0].srcdata['feat']
                        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                        pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
                        score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
                        loss = F.binary_cross_entropy_with_logits(score, labels)
                        opt.zero_grad()
                        loss.backward()
                        batch_losses.append(loss.item())
                        opt.step()    
                print("Epoch:{}, Batch Loss:{} ".format(epoch, np.mean(batch_losses)))
                
                valid_loss, valid_ap, valid_auc = [] , [], [] 
                model.eval()  
                for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(val_dataloader):
                        x = blocks[0].srcdata['feat']
                        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                        pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
                        score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
                        loss = F.binary_cross_entropy_with_logits(score, labels)
                        roc_score = roc_auc_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
                        ap_score = average_precision_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
                        valid_loss.append(loss.item())
                        valid_auc.append(roc_score)
                        valid_ap.append(ap_score)
                print("Epoch:{}, Valid Loss:{}, Valid AUC:{}, Valid AP:{}".format(epoch, np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_ap)))
                
                # ap.append(np.mean(valid_ap))
                # auc.append(np.mean(valid_auc))
                save_state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                }
                
                backup_fpath = os.path.join(p + 'checkpoints/from_{}.pt'.format(epoch)) 
                print("saving checkpoint for epoch:{} to:{}".format(epoch, backup_fpath))
                torch.save(save_state, backup_fpath) 

def get_emb(epoch, p): 
        #Build graph 
        g, reverse_eids, train_eids, val_eids = build()

        #Create samplers 

        device = 'cuda'
        g = g.to(device)
        in_size = g.ndata['feat'].shape[1]
        model = SAGE(in_size, 2560).to(device)

        checkpoint = p+'checkpoints/from_{}.pt'.format(epoch)
        
        model_state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(model_state['model_state'])
        all_embeddings = model.inference(g, device, 32)
        mask = g.ndata['_TYPE'].bool()
        playlist_embeddings = all_embeddings[~mask]
        track_embeddings = all_embeddings[mask]
        print("Saving embeddings of size:{}(track), {}(playlist)".format(track_embeddings.shape, playlist_embeddings.shape))

        pickle.dump(track_embeddings, open(p+'embeddings/track_emb_{}.pkl'.format(epoch), "wb"))
        pickle.dump(playlist_embeddings, open(p+'embeddings/playlist_emb_{}.pkl'.format(epoch), "wb"))

def make_recs(epoch, p):
        #Params  
        gen_amount = 10 
        k = 500 
        #Data paths
        train_path = f'{dataset_path}/datasets/small_100_10/train_val.pkl'
        test_path = f'{dataset_path}/datasets/small_100_10/test.pkl'
        embed_path = p+'embeddings/track_emb_{}.pkl'.format(epoch)
        rec_path = p+'recs/cosine_recs_{}.pkl'.format(epoch)
        #Load data 
        data = pickle.load(open(train_path, "rb")) 
        test_data = pickle.load(open(test_path, "rb")) 
        df_items = data['df_track']
        df_users = data['df_playlist_info']
        df_interactions = data['df_playlist']
        #Load embds 
        track_embeddings = pickle.load(open(embed_path, "rb"))
        track_embeddings = track_embeddings.detach()
        print("***loaded track embedding of size:{}***".format(track_embeddings.size()))

        #Generate recs 
        recommended_tracks = [] 
        for pid in tqdm(test_data.pid.unique()): 
                associated_tracks = test_data[test_data.pid == pid]['tid'].tolist()
                associated_tracks = associated_tracks[:gen_amount]
                playlist_embedding = torch.mean(track_embeddings[associated_tracks], axis=0).reshape(1, -1)
                sim_values = torch.Tensor(cosine_similarity(playlist_embedding, track_embeddings))
                recs = torch.topk(sim_values, k)[1].tolist() 
                rec_tracks = df_items[df_items.tid.isin(recs)].tid.tolist() 
                recommended_tracks.append(recs) 
        rec_df = pd.DataFrame({'pid': test_data.pid.unique(), 'recs': recommended_tracks})
        pickle.dump(rec_df, open(rec_path, "wb"))
           

