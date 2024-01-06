import numpy as np
import pickle
from tqdm import tqdm 
import torch 
import pandas as pd
from torch.nn.functional import cosine_similarity
import ipdb
from scipy.sparse import csr_array 
import time 


def MaxFlow_sp(C, U, I, track_idx, plist_idx):
    s, t = 0, C.shape[0] -1 
    n = C.shape[0] # C is the capacity matrix
    F = csr_array((n, n)) #.toarray() 

    # the residual capacity from u to v is C[u][v] - F[u][v]
    height = np.zeros((n)) #label of node 
    height[0] = U + I + 2
    height[track_idx] = 2
    height[plist_idx] = 1
    
    excess = np.zeros(n) # flow into node minus flow from node
    seen = np.zeros(n) # neighbours seen since last relabel
    
    nodelist =  list(range(1, n))  
    
    height = height.astype(int)
    seen = seen.astype(int)
    excess = excess.astype(int)
    

    #push operation
    def push(u, v):
#         ipdb.set_trace()
        send = min(excess[u], C[[u],[v]] - F[[u],[v]])
        F[[u],[v]] += send #--> needs to change 
        F[[v],[u]] -= send #--> needs to change 
        excess[u] -= send
        excess[v] += send

    #relabel operation
    def relabel(u):
        # find smallest new height making a push possible,
        # if such a push is possible at all
        min_height = np.iinfo(np.int32).max #float('inf')
        for v in range(n):
            if C[[u],[v]] - F[[u],[v]] > 0: #--> needs to change 
                min_height = min(min_height, height[v])
                height[u] = min_height + 1

    def discharge(u):
         while excess[u] > 0:
            if seen[u] < n: # check next neighbour
                v = seen[u]
                if C[[u],[v]] - F[[u],[v]] > 0 and height[u] > height[v]: #-->needs to change 
                     push(u, v)
                else:
                     seen[u] += 1
            else: # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0


    excess[s] = np.iinfo(np.int32).max # send as much flow as possible to neighbours of source
    
    for v in range(1,n):
        push(s, v)
 
    p = 0
    while p < len(nodelist):
        u = nodelist[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            nodelist.insert(0, nodelist.pop(p)) # move to front of list
            p = 0 # start from front of list
        else:
            p += 1
#     ipdb.set_trace()
    return np.sum(F[[s], :]), height



def MaxFlow(C, U, I, track_idx, plist_idx):

    s, t = 0, len(C)-1
    n = len(C) # C is the capacity matrix
    F = np.zeros((n,n)) #convert to int 

    # the residual capacity from u to v is C[u][v] - F[u][v]
    height = np.zeros((n)) #label of node 
    height[0] = U + I + 2
    height[track_idx] = 2
    height[plist_idx] = 1
    
    excess = np.zeros(n) # flow into node minus flow from node
    seen = np.zeros(n) # neighbours seen since last relabel
    
    nodelist =  list(range(1, n))  #track_idx.copy() #list(range(1, n)) 
    
    height = height.astype(int)
    seen = seen.astype(int)
    excess = excess.astype(int)
    

    #push operation
    def push(u, v):
        send = min(excess[u], C[u][v] - F[u][v])
        F[u][v] += send
        F[v][u] -= send
        excess[u] -= send
        excess[v] += send

    #relabel operation
    def relabel(u):
        # find smallest new height making a push possible,
        # if such a push is possible at all
        min_height = np.iinfo(np.int32).max #float('inf')
        for v in range(n):
            if C[u][v] - F[u][v] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1

    def discharge(u):
         while excess[u] > 0:
            if seen[u] < n: # check next neighbour
                v = seen[u]
                if C[u][v] - F[u][v] > 0 and height[u] > height[v]:
                     push(u, v)
                else:
                     seen[u] += 1
            else: # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0
 
#     height[s] = n   # longest path from source to sink is less than n long

    excess[s] = np.iinfo(np.int32).max #float("inf") # send as much flow as possible to neighbours of source
    
    #Preflow (?) --> changed this from range(n)
    for v in range(1,n):
        push(s, v)
    
    print("preflow complete")
    print("running PushRelabel")
    p = 0
    while p < len(nodelist):
        u = nodelist[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            nodelist.insert(0, nodelist.pop(p)) # move to front of list
            p = 0 # start from front of list
        else:
            p += 1
    return sum(F[s]), height

def calculate_weights_sparse(rec_df, lmbda, num_tracks, num_plists, plist_idx, track_idx , global_rel, verbose=False): 
    nodes = num_plists + num_tracks + 2
    rows = [] 
    cols = [] 
    vals = [] 
    for plist_num in range(num_plists): 
        u_idx = plist_idx[plist_num]
        pid = rec_df.pid.unique()[plist_num]
        row = rec_df[rec_df.pid == pid]
        tracks = row.recs.to_list() #
        for rank_ui in range(len(tracks)): 
            tid = tracks[rank_ui]
            t_idx = track_idx[tid]
            try: 
                weight = lmbda * rank_ui + (1-lmbda) * global_rel[tid]
                u = u_idx 
                i = t_idx
                rows.extend([u,i])
                cols.extend([i,u])
                vals.extend([weight,weight])
                if verbose: 
                    print("p:{}, t:{}, p_idx:{}, t_idx:{}, rank:{}, degree:{}, weight:{} " .format(pid, tid, u,i, rank_ui, global_rel[tid], weight)) 
            except Exception as e: 
                print("ERROR", e)
                continue
    #Sink/source too 
    # ipdb.set_trace()
    cap = np.sum(vals)
    C_I, C_U = int(cap/num_tracks), int(cap/num_plists) 
    source_val = np.min([C_I/np.gcd(C_I, C_U), C_U/np.gcd(C_I, C_U)])
    sink_val = C_I/np.gcd(C_I, C_U)

    rows.extend([0]*(len(track_idx))) #adding source 
    cols.extend([t for t in track_idx])
    vals.extend([source_val]*(len(track_idx)))

    
    rows.extend([u for u in plist_idx]) #adding sink 
    cols.extend([nodes-1]*(len(plist_idx))) 
    vals.extend([sink_val]*(len(plist_idx)))

    print(len(rows), len(vals), len(vals)) 
    #set source 
    # W[0, track_idx] = source_val
    
    # W[-1, plist_idx] = sink_val
    W_sparse = csr_array((vals, (rows, cols)), shape=(nodes,nodes)) 
    return W_sparse, np.sum(vals)

def calculate_weights(rec_df, lmbda, num_tracks, num_plists, plist_idx, track_idx , global_rel, verbose=False):
    nodes = num_plists + num_tracks + 2
    # ipdb.set_trace()
    W = np.zeros((nodes, nodes))
    for plist_num in range(num_plists): 
        u_idx = plist_idx[plist_num]
        pid = rec_df.pid.unique()[plist_num]
        row = rec_df[rec_df.pid == pid]
        tracks = row.recs.to_list() #
        for rank_ui in range(len(tracks)): 
            tid = tracks[rank_ui]
            t_idx = track_idx[tid]
            try: 
                weight = lmbda * rank_ui + (1-lmbda) * global_rel[tid]
                # u = u_idx + 1
                # i = num_plists + track
                u = u_idx 
                i = t_idx
                W[u][i] = W[i][u] = weight
                if verbose: 
                    print("p:{}, t:{}, p_idx:{}, t_idx:{}, rank:{}, degree:{}, weight:{} " .format(pid, tid, u,i, rank_ui, global_rel[tid], weight)) 
            except Exception as e: 
                print(e)
                continue 
    return W, np.sum(W)

def add_source_sink(W, cap, num_tracks, num_plists, plist_idx, track_idx ):
    C_I, C_U = int(cap/num_tracks), int(cap/num_plists) 
    source_val = np.min([C_I/np.gcd(C_I, C_U), C_U/np.gcd(C_I, C_U)])
    sink_val = C_I/np.gcd(C_I, C_U)
   
    #set source 
    W[0, track_idx] = source_val
    #set sink
    W[plist_idx, -1] = sink_val
    return W, plist_idx, track_idx,num_tracks, num_plists

def build_graph(exploded, lmbda, sparse=True): 
    global_rel = exploded.groupby('recs')['pid'].apply(len).to_dict()
    num_plists, num_tracks = len(exploded.pid.unique()), len(exploded.recs.unique()) 
    
    track_idx = list(range(1, num_tracks+1)) #list(range(plist_idx[-1]+1, plist_idx[-1]+1+num_tracks+1)) 
    plist_idx = list(range(track_idx[-1]+1, track_idx[-1]+num_plists+1)) #list(range(1, num_plists+1))
    if sparse: 
        W, cap = calculate_weights_sparse(exploded, lmbda, num_tracks, num_plists, plist_idx,track_idx ,  global_rel)
    else: 
        W_raw,cap = calculate_weights(exploded, lmbda, num_tracks, num_plists, plist_idx,track_idx ,  global_rel)
        W, plist_idx, track_idx, num_tracks, num_plists = add_source_sink(W_raw, cap, num_tracks, num_plists, plist_idx, track_idx)
    return W, plist_idx, track_idx, num_tracks, num_plists

def update_G(u_c, exploded): 
    # print(u_c)
    # ipdb.set_trace() 
    for pid, tid in u_c:
        idx = exploded[((exploded['pid'] == pid) & (exploded['recs'] == tid))].index
        exploded.drop(idx, inplace=True)
    exploded = exploded.reset_index(drop=True)
    # print(exploded)
    return exploded, len(exploded.pid.unique()), len(exploded.recs.unique())

def update_G2(u_c, W): 
    for pid, tid in u_c:
        W[pid][tid] = W[tid][pid] = 0 
        W[0][tid] = W[tid][0]= 0 
        # W[-1][pid] = W[pid][-1] = 0 
    return W 

def generate_candidates(exploded, lmbda=0.2, verbose=False, sparse=False): 
    candidates = [] 
    counter = 0  
    # exploded = rec_df.explode('recs').reset_index() #dataframe of the form pid, tid 
    run = True 
    G, plist_idx, track_idx, num_tracks, num_plists =  build_graph(exploded, lmbda, sparse=sparse)
    # ipdb.set_trace() 
    while True: 
        # print(G)
        b = time.time()
        if sparse: 
            _, I_c = MaxFlow_sp(G, num_plists, num_tracks, track_idx, plist_idx)
        else: 
            _, I_c = MaxFlow(G, num_plists, num_tracks, track_idx, plist_idx)
        a = time.time() 
        if verbose: 
            print("labels", I_c, "took {} sec to complete run".format(a-b))  
        u_c = [] 
        for t in range(num_tracks):
            us = exploded[exploded.recs == t].pid.unique() 
            t_idx = track_idx[t] #is this correct? 
            if I_c[t_idx] >= num_plists + num_tracks + 2: 
                u_c.extend([(u, t) for u in us])
        if len(u_c) == 0:
            print("THEIR BREAK")
            break 
        candidates.extend(u_c)
        counter += 1 
        updated_rec,  num_plists, num_tracks =  update_G(u_c, exploded)
        if verbose: 
            print(counter, candidates)
            print(updated_rec, num_tracks, num_plists)
        G = update_G2(u_c, G)
        if np.sum(G) == 0:
            if verbose: 
                print("BREAK1")
            break 
        if num_tracks == 0 or num_plists == 0: 
            if verbose: 
                print("BREAK2")
            break 
        # G, plist_idx, track_idx, num_tracks, num_plists =  build_graph(updated_rec, lmbda)
    return candidates 

def make_rec_list(rec_df, candidates, beta=0.2, n=100): 
    # ipdb.set_trace()
    candidate_df = pd.DataFrame({'pid':[p for p,t in candidates], 'recs':[t for p,t in candidates]}).sort_values('pid').reset_index(drop=True)
    # candidate_df = candidate_df.
    candidate_vis = candidate_df.groupby('recs')['pid'].count().reset_index(name='vis').sort_values(['vis'], ascending=False) 
    candidates_by_vis = pd.merge(candidate_df, candidate_vis).sort_values(['pid', 'vis'], ascending=False)
    fair_df = candidates_by_vis.groupby('pid')['recs'].apply(list).reset_index(name='recs')
    blended_recs = [] 
    for pid in rec_df.pid.unique(): 
        og_recs = rec_df[rec_df.pid == pid].recs.tolist()[0][:n]
        fair_recs = fair_df[fair_df.pid == pid].recs.tolist()[0]
        keep_idx = min(int(len(og_recs) * beta), len(fair_recs))  
        fair_recs = [r for r in fair_recs if r not in og_recs]
        if len(fair_recs) >= keep_idx: 
            final_recs = og_recs[:-keep_idx] + fair_recs[:keep_idx]
        else: 
            final_recs = og_recs
        blended_recs.append(final_recs)
    final_df = pd.DataFrame({'pid': rec_df.pid.unique(), 'recs': blended_recs})
    return final_df 




def push_relabel_main(p, map_idx=True, beta=0.2, n=100, lmbda = 0.2, output_path=None, sparse=False): 
    testing = False 
    rec_df = pickle.load(open(p, "rb"))
    exploded = rec_df.explode('recs')
    num_plists = len(rec_df.pid.unique()) 
    num_tracks = len(exploded.recs.unique())
    biggest_plist_idx = max(exploded.pid.unique())
    biggest_track_idx = max(exploded.recs.unique())
    
    print(num_plists, num_tracks, biggest_plist_idx, biggest_track_idx)
    if map_idx: 
        pid_dict = dict(zip(rec_df.pid.unique(), list(range(num_plists))))
        tid_dict = dict(zip(exploded.recs.unique(), list(range(num_tracks))))
        exploded['pid'] = exploded['pid'].map(lambda x: pid_dict[x])
        exploded['recs'] = exploded['recs'].map(lambda x: tid_dict[x])
        num_plists = len(rec_df.pid.unique()) 
        num_tracks = len(exploded.recs.unique())
        biggest_plist_idx = max(exploded.pid.unique())
        biggest_track_idx = max(exploded.recs.unique())
        print("MAPPED IDX", num_plists, num_tracks, biggest_plist_idx, biggest_track_idx)
    if testing: 
        b = time.time()
        candidates_sp = generate_candidates(exploded, lmbda=lmbda, verbose=False, sparse=True)
        a = time.time() 
        print("generated candidates_sp in :{}".format(a-b))
        
    b = time.time()
    candidates = generate_candidates(exploded, lmbda=lmbda, verbose=False, sparse=sparse)
    a = time.time() 
    print("generated candidates in {}".format(a-b))
    
    if map_idx:
        reverse_pid_dict = {v:k for k,v in pid_dict.items()}
        reverse_tid_dict = {v:k for k,v in tid_dict.items()}
        candidates = [(reverse_pid_dict[p], reverse_tid_dict[t]) for p,t in candidates]
        if testing: 
            candidates_sp = [(reverse_pid_dict[p], reverse_tid_dict[t]) for p,t in candidates_sp]

    final_df = make_rec_list(rec_df, candidates, beta=beta, n=n)
    if testing: 
        final_df_sp = make_rec_list(rec_df, candidates_sp, beta=beta, n=n)

        print(final_df.equal(final_df_sp)) 
    # print(final_df)
    print(final_df.equals(rec_df)) 


    if output_path: 
        # output_file = output_path+ "PL_Benchmark_B_{}.pkl".format(beta, n) 
        pickle.dump(final_df, open(output_path, "wb"))



def sparse_tests(rec_df, lmbda): 
    ipdb.set_trace()
    lmbda = lmbda
    exploded = rec_df.explode('recs').reset_index(drop=True)
    global_rel = exploded.groupby('recs')['pid'].apply(len).to_dict()
    num_plists, num_tracks = len(exploded.pid.unique()), len(exploded.recs.unique()) 
    track_idx = list(range(1, num_tracks+1)) #list(range(plist_idx[-1]+1, plist_idx[-1]+1+num_tracks+1)) 
    plist_idx = list(range(track_idx[-1]+1, track_idx[-1]+num_plists+1)) #list(range(1, num_plists+1))
    W_raw_sp,  cap = calculate_weights_sparse(exploded, lmbda, num_tracks, num_plists, plist_idx,track_idx ,  global_rel)
   
    _, I_c = MaxFlow_sp(W_raw_sp, num_plists, num_tracks, track_idx, plist_idx)
    print(I_c)
