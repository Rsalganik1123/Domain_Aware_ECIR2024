import torch 
from tqdm import tqdm 
import pickle 
from torch.nn.functional import cosine_similarity
from numpy.linalg import norm 
import numpy as np 
# from fairness.fairness_utils import *  #only works is launched as src2/fairness/fairness_audit.py 


def avg_ndcg(x_corresponding, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1) #.cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding[:, :top_k]) - 1 #.cuda() 
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1) #.cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def generate_emb(model, dataloader): 
    '''
    Calculates the entire block of |V| x |V| embeddings 
    '''
    model = model.eval()
    model = model.cuda()
    #GENERATE REPRESENTATIONS 
    embeddings = [] 
    with torch.no_grad():
        for blocks in tqdm(iter(dataloader)):
            for i in range(len(blocks)): 
                blocks[i] = blocks[i].to('cuda')
            emb = model.get_repr(blocks)
            embeddings.append(emb.cpu().numpy()) 
    embeddings = torch.concatenate(embeddings)
    return embeddings 

def load_emb(path, specifier=None): 
    '''
    Load pre-computed embeddings 
    '''
    data = pickle.load(open(path, "rb")) 
    if specifier: 
        data = data[specifier]
    return data 

def load_feat(path):
    data = pickle.load(open(path, "rb"))
    df_items = data['df_track'][['tid', 'arid', 'followers', 'popularity', 'danceability', 'energy',
       'loudness', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo','followers_10cat', 'popularity_10cat']]
    return df_items.to_numpy()  

def sim(v, all_v, version=0): 
    if version == 1: 
        r = cosine_similarity(v, all_v)
    elif version == 2: 
        all_v = all_v/norm(all_v, axis=1)
        a = a/norm(playlist_embedding)
        r = 5 * (torch.mm(all_v, a.transpose(0, 1)) + 1)
    else: 
        r = 5 * cosine_similarity(v, all_v) + 1
    return r

def isolate_topk(embeddings, k, batch=1000, version=0): 
    '''
    For each node, get the top k most similar values and their idx 
    '''
    vals, idxs = [] , [] 
    embeddings = embeddings.cuda()
    if batch == 0: 
        batch = embeddings.shape[0]
    for idx in tqdm(range(0,batch)): #go over all embeddings  #embeddings.shape[0]
        e = embeddings[idx, :] #isolate one embeddings
        recs = sim(e, embeddings, version=version) #find all cosine similarities --> 1 x |V|
        val, idx = torch.topk(recs, k+1) #isolate top vals and idx
        vals.append(val)
        idxs.append(idx)
    vals = torch.stack(vals)
    idxs = torch.stack(idxs)
    return vals.cpu(), idxs.cpu()

def find_corresponding(idx, features, version=0): 
    '''
    For a y ordering, find the corresponding x values 
    '''
    x_corresponding = torch.zeros(idx.shape) 
    for i in range(idx.shape[0]): #for each nodes
        top_items = idx[i, :].int() - 1 #get the top most similar items (based on S_Y)
        my_item_feature = torch.tensor(features[i, :])
        top_item_features = torch.tensor(features[top_items, :])
        x_corresponding[i, :] = sim(my_item_feature , top_item_features,  version=version) #on the fly similarity calculation 
    return x_corresponding 

def format(vals, idxs): 
    '''
    '''
    vals = vals[:, 1:].float() 
    idxs = idxs[:, 1:].float() + 1
    return vals, idxs

def truncated_lambdas_computation(x_sorted_scores, x_sorted_idxs, y_sorted_scores, y_sorted_idx, x_corresponding, top_k):
    
    sigma_tuned = 1.0

    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    # x_corresponding = torch.zeros(x_sorted_scores.shape[0], x_sorted_scores.shape[1])

    # for i in range(x_corresponding.shape[0]):
    #     x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_sorted_scores.shape[0]):
        # if i >= 0.6 * y_similarity.shape[0]:
        #     break
        idcg = idcg_computation(x_sorted_scores[i, :], top_k)
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(x_corresponding[i, :], j, k, idcg, top_k)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])
    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   # 本来是 -


    # mid = torch.zeros_like(x_similarity)
    # the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    # the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    # the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    # mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return x_sorted_scores, y_sorted_idxs, x_corresponding

def run_audit(): 
    emb = load_emb('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/music+genre+meta_focal_norm_contig/track_emb/embeddings_as_array_fullg.pkl')
    feat = load_feat('/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/complete_data_final_3way_with_emb_and_pos_contig.pkl')

    print("Loaded embeddings of size:{} and features of size:{}".format(emb.shape, feat.shape))

    y_vals, y_idxs = isolate_topk(torch.tensor(emb), 10, batch = 0, version=0)
    x_vals, x_idxs = isolate_topk(torch.tensor(feat), 10, batch = 0, version=0)
    y_vals, y_ranks = format(y_vals, y_idxs)

    x_corresponding = find_corresponding(y_ranks, feat)
    should_return = avg_ndcg(x_corresponding, x_vals, y_idxs, top_k=5)
    print(should_return)

run_audit()