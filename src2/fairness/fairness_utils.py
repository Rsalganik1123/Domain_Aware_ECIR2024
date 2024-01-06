import numpy as np
import scipy.sparse as sp
import torch
import pickle 
from tqdm import tqdm 


def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)).cuda()
    final = numerator / denominator

    return torch.sum(final)

def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator

    return torch.sum(final)

def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)

def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def simi(output):  # new_version
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)
    # print("similarity matrix shape", res.shape)
    return res

def chunk_simi(chunk, embs): 
    
    chunk_norm = chunk / chunk.norm(dim=1)[:, None]
    emb_norm = embs / embs.norm(dim=1)[:, None]
    res = 5 * (torch.mm(chunk_norm, emb_norm.transpose(0, 1)) + 1)
    # print("c", chunk_norm.shape, "e", emb_norm.shape, 'res', res.shape)
    return res 

def simi_mine(output):
    a_norm = output / output.norm(dim=1)[:, None]
    res = torch.mm(a_norm, a_norm.transpose(0,1))
    return res 

def isolate_features(cfg, blocks, sim_feat=None):
    # subgraph = dgl.merge(blocks)
    # d1_feats = ['tempo', 'liveness', 'instrumentalness', 'speechiness', 'loudness', 'acousticness']
    # d2_feats = ['genre']
    
    # d1_feats = cfg.MODEL.PINSAGE.PROJECTION.CONCAT
    # d2_feats = ['img_emb', 'track_name_emb', 'genres', 'genre']

    d1_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'CAT']
    d2_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'VEC']
    # print(sim_feat, d1_feats, d2_feats)
    if not sim_feat: 
        sim_feat = d1_feats + d2_feats
    feature_list = [] 
    # print("avilable features", blocks[-1].dstdata.keys())
    for k in sim_feat: 
        # print(k)
        if k in d1_feats: 
            feature = blocks[-1].dstdata[k].reshape(-1, 1).float()
            feature_list.append(feature)
        if k in d2_feats: 
            feature = blocks[-1].dstdata[k]
            
            feature_list.append(feature)
    features = torch.cat(feature_list, axis=1)
    
    # for k in d1_feats: 
    #     if 'genre' in k: continue 
    #     feature = blocks[-1].dstdata[k].reshape(-1, 1)
    #     feature_list.append(feature)
    # features = torch.cat(feature_list, dim=1)
    # for k in d2_feats: 
    #     try: 
    #         features = torch.cat([features, blocks[-1].dstdata[k]], dim=1)
    #     except: 
    #         continue 
        
    return features #.cpu()
    
def cutoff(mod_sim): 
    max_fill = torch.full_like(mod_sim, 10.0)
    cutoff_sim = torch.where(mod_sim > 10.0, max_fill, mod_sim)
    return cutoff_sim

def isolate_pop(blocks, pop_feat, mode, boost_val = 0.01): #boost=0.0, method='vanilla',
    if mode == 'boost1':  #try exponential or log to make the difference more smooth 
        #try using the focal loss to define the coefficient 
        pop_val = blocks[-1].dstdata[pop_feat].reshape(1, -1).float()
        pop_diff = pop_val[0, :].view(pop_val.shape[1], 1) - pop_val[0, :].float() 
        return pop_diff * torch.full_like(pop_diff, boost_val)
    if mode == 'boost2': 
        pop_val = blocks[-1].dstdata[pop_feat].reshape(1, -1).float()
        pop_diff = torch.abs(pop_val[0, :].view(pop_val.shape[1], 1) - pop_val[0, :].float()) 
        return pop_diff
        
    if mode == 'boost3': 
        pop_val = blocks[-1].dstdata[pop_feat].reshape(1, -1).float()
        pop_diff = torch.abs(pop_val[0, :].view(pop_val.shape[1], 1) - pop_val[0, :].float()) 
        return 1 - pop_diff 

    if mode == 'boost4': 
        pop_val = blocks[-1].dstdata[pop_feat].reshape(1, -1).float() + 1
        pop_diff = torch.abs((pop_val[0, :].view(pop_val.shape[1], 1) - pop_val[0, :].float()) / (10)) 
        return torch.abs(pop_diff) 

    # if mode == 'weighted': 
    #     pop_val = blocks[-1].dstdata[pop_feat].reshape(1, -1).float()
    #     return pop_val 
    if mode == 'weighted_old': 
        pop_val = blocks[-1].dstdata['log10_popcat'].reshape(1, -1).float() + 1 
        pop_diff = (pop_val[0, :].view(pop_val.shape[1], 1) - pop_val[0, :].float()) / (10) 
        return torch.abs(pop_diff) #* torch.full_like(pop_diff, boost_val)

    
def isolate_embeddings(model, blocks): 
    embeds = model.get_repr(blocks) #.cpu()
    # print("embeds shape", embeds.shape)
    return embeds


def lambdas_computation_old(cfg, x_similarity, y_similarity, top_k):
    # ipdb.set_trace()
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    sigma_tuned = cfg.FAIR.ALPHA
    length_of_k = top_k #k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0]) #cpu
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k) #.to('cuda')  #cpu 

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]:
            break
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


    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding


def lambdas_computation(cfg, x_similarity, y_similarity, top_k, weights = None):
    
    if weights != None: 
        # print("weighted example")
        x_similarity = x_similarity * weights 
    
    
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    sigma_tuned =  cfg.FAIR.ALPHA
    length_of_k = top_k 
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    
    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0]) #cpu
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k) #.to('cuda')  #cpu 

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]: 
            break
            
        idcg = idcg_computation(x_sorted_scores[i, :top_k+1], top_k)

        for j in range(top_k):
            for k in range(top_k):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(x_corresponding[i, :top_k+1], j, k, idcg, top_k)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])
    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   # 本来是 -


    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    
    # print(mid.dtype, the_x.dtype)
    
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding


def calc_fair_loss(cfg, model, blocks, sim_feat=None): 
    # ipdb.set_trace() 
    embeddings = isolate_embeddings(model, blocks)
    features = isolate_features(cfg, blocks, sim_feat=sim_feat)
    apriori_sim = simi(features)
    pred_sim = simi(embeddings)
    
    # if cfg.FAIR.NDCG_METHOD == 'weighted2':
    #     weights = isolate_pop(blocks, boost_val = cfg.FAIR.BOOST, pop_feat='log10_popcat', mode ='weighted2')
    #     # inv_weights = 1 - weights
    #     pred_sim = pred_sim  + weights
    #     lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(cfg, apriori_sim, pred_sim, cfg.FAIR.TOP_K)
    
    # elif cfg.FAIR.NDCG_METHOD == 'weighted':
    #     weights = isolate_pop(blocks, boost_val = 0.0, pop_feat=cfg.FAIR.POP_FEAT, mode ='weighted').reshape(-1, 1)
    #     # inv_weights = 1 - weights
    #     inv_weights = weights
    #     pop = inv_weights.view(inv_weights.shape[0], 1).repeat(1, inv_weights.shape[0]).float()
    #     lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(cfg, apriori_sim, pred_sim, cfg.FAIR.TOP_K, weights = pop)

    # elif cfg.FAIR.BOOST != 0.0 and : 
    #     boost_mat = isolate_pop(blocks, boost_val = cfg.FAIR.BOOST, pop_feat='popularity_10cat')
    #     pred_sim = pred_sim  + boost_mat

    
    if cfg.FAIR.NDCG_METHOD == 'boost1': #multiplies by inputed value 
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost1', boost_val=cfg.FAIR.BOOST)
        pred_sim = pred_sim  + pop_mat

    if cfg.FAIR.NDCG_METHOD == 'boost2': #abs(difference between bins)
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost2')
        pred_sim = pred_sim  + pop_mat

    if cfg.FAIR.NDCG_METHOD == 'boost3': #difference as probability 
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost3')
        pred_sim = pred_sim  + pop_mat

    if cfg.FAIR.NDCG_METHOD == 'boost4': #normalized version of v2
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost4')
        pred_sim = pred_sim  + pop_mat

    if cfg.FAIR.NDCG_METHOD == 'boost5': #cutoff version of v2 
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost2')
        pred_sim = pred_sim  + pop_mat
        pred_sim = cutoff(pred_sim)

    if cfg.FAIR.NDCG_METHOD == 'boost6': #boost2 but applied to apriori  
        pop_mat = isolate_pop(blocks, cfg.FAIR.POP_FEAT, 'boost2')
        apriori_sim = apriori_sim  + pop_mat
        # pred_sim = cutoff(pred_sim)

    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(cfg, apriori_sim, pred_sim, cfg.FAIR.TOP_K)

    assert lambdas.shape == pred_sim.shape
    
    should_return = avg_ndcg(x_corresponding.cuda(), apriori_sim, x_sorted_scores, y_sorted_idxs, top_k=cfg.FAIR.TOP_K)
    
    return should_return, pred_sim.to('cuda'), lambdas.to('cuda')

def calc_fair_loss_entireg(cfg, model, dataloader, graph, feat): 

    d1_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'CAT']
    d2_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'VEC']
    print("Overall fairness calculation")
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
    embeddings = np.concatenate(embeddings)
    feature_list = [] 
    for f in feat: 
        feature = graph.nodes['track'].data[f].to('cuda')
        if f in d1_feats: 
            feature = feature.reshape(-1, 1).float()
        feature_list.append(feature)
    # ipdb.set_trace()
    features = torch.cat(feature_list, axis=1)
    embeddings = torch.tensor(embeddings).to('cuda')
    
    apriori_sim = simi(features)
    pred_sim = simi(embeddings)
    
    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(cfg, apriori_sim, pred_sim, cfg.FAIR.TOP_K)
    assert lambdas.shape == pred_sim.shape
    should_return = avg_ndcg(x_corresponding.cuda(), apriori_sim, x_sorted_scores.cuda(), y_sorted_idxs.cuda(), top_k=cfg.FAIR.TOP_K)
    return should_return

def global_loss_TEST(cfg, track_emb): 
    # ipdb.set_trace()
    d1_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'CAT']
    d2_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'VEC']
    # print("Overall fairness calculation")
    
    features = chunk_features(cfg, pickle.load(open(cfg.DATASET.DATA_PATH, "rb"))[cfg.DATASET.ITEM_DF], cfg.FAIR.FEAT_SET).to('cuda')
    embeddings = track_emb.to('cuda')
    
    apriori_sim = simi(features)
    pred_sim = simi(embeddings)
    
    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(cfg, apriori_sim, pred_sim, cfg.FAIR.TOP_K)
    assert lambdas.shape == pred_sim.shape
    should_return = avg_ndcg(x_corresponding.cuda(), apriori_sim, x_sorted_scores.cuda(), y_sorted_idxs.cuda(), top_k=cfg.FAIR.TOP_K)
    return should_return

def chunk_features(cfg, features, feat): 
    d1_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'CAT']
    d2_feats = [i for [i, t] in cfg.DATASET.ITEM_FEATURES if t == 'VEC']
    feature_list = []  
    for f in feat: 
        feature = torch.tensor(features[f]) 
        if f in d1_feats: 
            feature = feature.reshape(-1, 1).float()
        feature_list.append(feature)
        
    iso_features = torch.cat(feature_list, axis=1)
    return iso_features

def chunked_y_simi_calculations(cfg, data_path=None, output_path = None, chunk_num = 1, chunk_size=100, track_emb=None): 
    simi_chunks = [] 
    k = cfg.FAIR.TOP_K
    grp = cfg.FAIR.FEAT_SET
    #load data
    if data_path: 
        track_emb = torch.tensor(pickle.load(open(data_path, "rb"))) 
    features = chunk_features(cfg, pickle.load(open(cfg.DATASET.DATA_PATH, "rb"))[cfg.DATASET.ITEM_DF], grp).cuda()  
    start = 0 
    ndcg = 0.0
    fairness = [] 
    
    for end in range(0 + chunk_size, track_emb.shape[0], chunk_size): 
        track_chunk = track_emb[start:end, :]
        feature_chunk = features[start:end, :]
        y_simi = chunk_simi(track_chunk, track_emb)
        x_simi = chunk_simi(feature_chunk, features)
        # y_sorted_scores, y_sorted_idxs = torch.topk(y_simi, k-1, dim=1, largest=True, sorted=True)
        # x_sorted_scores, x_sorted_idxs = torch.topk(x_simi, k-1, dim=1, largest=True, sorted=True)
        # x_sorted_scores = x_sorted_scores.cuda() 
        # y_ranks = 1 + y_sorted_idxs.float() 

        (x_sorted_scores, x_sorted_idxs) = x_simi.sort(dim=1, descending=True)
        (y_sorted_scores, y_sorted_idxs) = y_simi.sort(dim=1, descending=True)
        y_ranks = torch.zeros(y_simi.shape[0], y_simi.shape[0])
        the_row = torch.arange(y_simi.shape[0]).view(y_simi.shape[0], 1).repeat(1, y_simi.shape[0])
        y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_simi.shape[1]).repeat(y_simi.shape[0], 1).float()


        x_corresponding = torch.zeros(x_simi.shape[0], k-1).cuda()
        for i in range(x_corresponding.shape[0]):
            x_corresponding[i, :] = x_simi[i, y_sorted_idxs[i, :]]

        ndcg = avg_ndcg(x_corresponding, x_simi, x_sorted_scores, y_ranks, k)
        fairness.append((ndcg, track_chunk.shape[0]))
        start = end
    if end < track_emb.shape[0]:
        remainder = track_emb.shape[0] - end
        track_chunk = track_emb[-remainder:, :]
        feature_chunk = features[-remainder:, :]
        y_simi = chunk_simi(track_chunk, track_emb)
        x_simi = chunk_simi(feature_chunk, features)
        # y_sorted_scores, y_sorted_idxs = torch.topk(y_simi, k, dim=1, largest=True, sorted=True)
        # x_sorted_scores, x_sorted_idxs = torch.topk(x_simi, k, dim=1, largest=True, sorted=True)
        # x_sorted_scores = x_sorted_scores.cuda() 
        # y_ranks = 1 + y_sorted_idxs.float() 

        (x_sorted_scores, x_sorted_idxs) = x_simi.sort(dim=1, descending=True)
        (y_sorted_scores, y_sorted_idxs) = y_simi.sort(dim=1, descending=True)
        y_ranks = torch.zeros(y_simi.shape[0], y_simi.shape[1])
        the_row = torch.arange(y_simi.shape[0]).view(y_simi.shape[0], 1).repeat(1, y_simi.shape[0])
        y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_simi.shape[1]).repeat(y_simi.shape[0], 1).float()

        x_corresponding = torch.zeros(x_simi.shape[0], k).cuda()
        for i in range(x_corresponding.shape[0]):
            x_corresponding[i, :] = x_simi[i, y_sorted_idxs[i, :]]

        ndcg = avg_ndcg(x_corresponding, x_simi, x_sorted_scores, y_ranks, k)
        fairness.append((ndcg, track_chunk.shape[0]))

    total_sum = np.sum([k*v for (k,v) in fairness])
    total_div = np.sum([v for (k,v) in fairness])
    return total_sum/total_div

def test():  #weighted average 
    # ipdb.set_trace() 
    vals = np.array([[1,3,4,5,6,7,8]]) 
    g = np.mean(vals)
    start = 0 
    chunk_size = 2
    c = [] 
    for end in tqdm(range(1, (vals.shape[1]//chunk_size)+2)): 
        section = vals[start*chunk_size:end*chunk_size, :]
        v = np.mean(section)
        c.append(v* (section.shape[0]/len(vals))) 
    print(g, np.sum(c), g == np.sum(c))

