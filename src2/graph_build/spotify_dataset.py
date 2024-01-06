import pickle
import torch
import numpy as np
from src2.graph_build.data_load import DATASET_REGISTRY

from src2.utils.build_utils.dgl_builder import PandasGraphBuilder
import dgl
 
# print("DATASET REGISTRY", DATASET_REGISTRY)

@DATASET_REGISTRY.register('SMALL')
def build_graph(cfg): 
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    print("features available: {}".format(df_items.columns))
    train_indices = all_data['train_indices']
    
    train_user_ids = all_data['train_pids']
    val_user_ids = all_data['val_pids']
    test_user_ids = all_data['test_pids']
    
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()
    
    # pids = torch.tensor(np.concatenate([train_user_ids, val_user_ids, test_user_ids]))
    # tids = torch.tensor(np.concatenate([train_item_ids, val_item_ids, test_item_ids])).unique()
    g.nodes[cfg_data.USER].data['id'] = torch.arange(g.number_of_nodes(cfg_data.USER))
    g.nodes[cfg_data.ITEM].data['id'] = torch.arange(g.number_of_nodes(cfg_data.ITEM))
    # g.nodes[cfg_data.USER].data['id'] = pids
    # g.nodes[cfg_data.ITEM].data['id'] = tids
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        if feature_type == 'CAT':
            values = torch.LongTensor(df_items[key].values)
        else:
            values = torch.tensor(np.asarray(list(df_items[key].values))).float()
        g.nodes[cfg_data.ITEM].data[key] = values
    train_g = build_train_graph(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, [train_user_ids, val_user_ids, test_user_ids]

@DATASET_REGISTRY.register('DEFAULT')
def load_entire_dataset(cfg): 
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']

    print("features available: {}".format(df_items.columns))
    
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()

    # g.nodes[cfg_data.USER].data['pid'] = torch.tensor(df_users.pid.unique()) 
    # g.nodes[cfg_data.ITEM].data['tid'] = torch.tensor(df_items.tid.unique()) 
    
    
    g.nodes[cfg_data.USER].data['pid'] = torch.arange(g.number_of_nodes(cfg_data.USER)) #id
    g.nodes[cfg_data.ITEM].data['tid'] = torch.arange(g.number_of_nodes(cfg_data.ITEM)) #id
    
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        if feature_type == 'CAT':
            values = torch.LongTensor(df_items[key].values)
        else:
            values = torch.tensor(np.asarray(list(df_items[key].values))).float()
        g.nodes[cfg_data.ITEM].data[key] = values
    train_g = build_train_graph(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    val_g = build_train_graph(g, val_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, val_g

@DATASET_REGISTRY.register('NO_ISOLATE')
def load_entire_dataset(cfg): 
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']

    val_data = df_interactions.loc[val_indices]
    print("features available: {}".format(df_items.columns))
    
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()

    # g.nodes[cfg_data.USER].data['pid'] = torch.tensor(df_users.pid.unique()) 
    # g.nodes[cfg_data.ITEM].data['tid'] = torch.tensor(df_items.tid.unique()) 
    
    
    g.nodes[cfg_data.USER].data['pid'] = torch.arange(g.number_of_nodes(cfg_data.USER)) #id
    g.nodes[cfg_data.ITEM].data['tid'] = torch.arange(g.number_of_nodes(cfg_data.ITEM)) #id
    
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        try: 
            if feature_type == 'CAT':
                values = torch.LongTensor(df_items[key].values)
            if feature_type == 'FLT':
                values = torch.FloatTensor(df_items[key].values)
            if feature_type == 'VEC':
                values = torch.tensor(np.asarray(list(df_items[key].values))).float()
            g.nodes[cfg_data.ITEM].data[key] = values
        except: 
            print("couldn't load key:", key)
    print("Node features", g.nodes[cfg_data.ITEM].data.keys())
     
    train_g = build_train_graph2(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    val_g = build_train_graph2(g, val_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, val_g


@DATASET_REGISTRY.register('NO_ISOLATE2')
def load_entire_dataset(cfg): 
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']

    val_data = df_interactions.loc[val_indices]
    print("features available: {}".format(df_items.columns))
    
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()

    # g.nodes[cfg_data.USER].data['pid'] = torch.tensor(df_users.pid.unique()) 
    # g.nodes[cfg_data.ITEM].data['tid'] = torch.tensor(df_items.tid.unique()) 
    
    
    g.nodes[cfg_data.USER].data['pid'] = torch.arange(g.number_of_nodes(cfg_data.USER)) #id
    g.nodes[cfg_data.ITEM].data['tid'] = torch.arange(g.number_of_nodes(cfg_data.ITEM)) #id
    
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        try: 
            if feature_type == 'CAT':
                values = torch.LongTensor(df_items[key].values)
            if feature_type == 'FLT':
                values = torch.FloatTensor(df_items[key].values)
            if feature_type == 'VEC':
                values = torch.tensor(np.asarray(list(df_items[key].values))).float()
            g.nodes[cfg_data.ITEM].data[key] = values
        except: 
            print("couldn't load key:", key)
    print("Node features", g.nodes[cfg_data.ITEM].data.keys())
     
    train_g = build_train_graph2(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    val_g = build_train_graph2(g, val_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, val_g, val_data

@DATASET_REGISTRY.register('PREMADE_EMB')
def load_entire_dataset(cfg): 
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    print("loaded dataset: {} with fields:{}".format(cfg_data.NAME, all_data.keys()))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data['train_indices']
    val_indices = all_data['val_indices']

    print("features available: {}".format(df_items.columns))
    
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()

    # g.nodes[cfg_data.USER].data['pid'] = torch.tensor(df_users.pid.unique()) 
    # g.nodes[cfg_data.ITEM].data['tid'] = torch.tensor(df_items.tid.unique()) 
    
    
    g.nodes[cfg_data.USER].data['pid'] = torch.arange(g.number_of_nodes(cfg_data.USER)) #id
    g.nodes[cfg_data.ITEM].data['tid'] = torch.arange(g.number_of_nodes(cfg_data.ITEM)) #id
    
    features = cfg_data.ITEM_FEATURES
    feats = [] 
    for key, feature_type in features:
        if feature_type == 'CAT':
            values = torch.LongTensor(df_items[key].values).reshape(-1, 1)
        else:
            values = torch.tensor(np.asarray(list(df_items[key].values))).float()
        g.nodes[cfg_data.ITEM].data[key] = values
        if key in cfg.MODEL.PINSAGE.PROJECTION.FEATURES:
            feats.append(values)

    track_features = torch.concat(feats, axis=1)
    g.nodes[cfg_data.ITEM].data['feats'] = track_features 

    print("Node features", g.nodes[cfg_data.ITEM].data.keys())
     
    train_g = build_subgraph(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    val_g = build_subgraph(g, val_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, val_g

@DATASET_REGISTRY.register('SPOTIFY_MUSIC')
def build_spotify_graphs(cfg):
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data[cfg_data.TRAIN_INDICES]
    train_user_ids = all_data['train_user_ids']
    val_user_ids = all_data['val_user_ids']
    test_user_ids = all_data['test_user_ids']
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()
    g.nodes[cfg_data.USER].data['id'] = torch.arange(g.number_of_nodes(cfg_data.USER))
    g.nodes[cfg_data.ITEM].data['id'] = torch.arange(g.number_of_nodes(cfg_data.ITEM))
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        if feature_type == 'CAT':
            values = torch.LongTensor(df_items[key].values)
        else:
            values = torch.tensor(np.asarray(list(df_items[key].values))).float()
        g.nodes[cfg_data.ITEM].data[key] = values
    train_g = build_train_graph(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, [train_user_ids, val_user_ids, test_user_ids]

def build_subgraph(g, indices, etype, etype_rev): 
    sub_g = g.edge_subgraph(
        {etype: indices, etype_rev: indices})

    for ntype in sub_g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            sub_g.nodes[ntype].data[col] = data[sub_g.nodes[ntype].data[dgl.NID]]
            

    for etype in sub_g.etypes:
        for col, data in g.edges[etype].data.items():
            sub_g.edges[etype].data[col] = data[sub_g.edges[etype].data[dgl.EID]]

    return sub_g

def build_train_graph(g, train_indices, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        relabel_nodes=False)

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g

def build_train_graph2(g, train_indices, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices})

    for ntype in train_g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            isolate_data = data[train_g.nodes(ntype)]
            train_g.nodes[ntype].data[col] = data[train_g.nodes[ntype].data[dgl.NID]]
            # train_g.nodes[ntype].data[col] = isolate_data

    for etype in train_g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]


    return train_g
