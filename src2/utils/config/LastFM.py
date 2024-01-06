from fvcore.common.config import CfgNode

#Initializing the CFG
_C = CfgNode()
_C.seed = 47 #Seed used throughout random sampling process 

#RUN SETTINGS 
_C.RUN = CfgNode()
_C.RUN.TESTING = False #If True: checkpoints aren't saved 
_C.RUN.PRETRAINED = False #If True: need to provide pretrained path for loading model weights 
_C.RUN.EARLY_STOPPING = False #If True: early stopping is initialized
_C.RUN.GLOBAL_FAIR_CALC = False #If True: Global REDRESS fairness calculations are done between utlity and fairness modes. Note: intractable for fullsized dataset 

#DATASET
_C.DATASET = CfgNode()
_C.DATASET.NAME = 'DEFAULT'
_C.DATASET.DATA_PATH = ''
_C.DATASET.USER_DF = 'df_playlist_info'
_C.DATASET.ITEM_DF = 'df_track'
_C.DATASET.INTERACTION_DF = 'df_playlist'
_C.DATASET.ITEM = 'track'
_C.DATASET.USER = 'playlist'
_C.DATASET.USER_ITEM_EDGE = 'contains'
_C.DATASET.ITEM_USER_EDGE = 'contained_by'

_C.DATASET.USER_ID = 'pid'
_C.DATASET.ITEM_ID = 'tid'
#Feature categories: CAT means scalar value, VEC means vector value 
_C.DATASET.ITEM_FEATURES = [
    ['tid', 'CAT'],
    ['danceability_10cat', 'CAT'],
    ['energy_10cat', 'CAT'],
    ['loudness_10cat', 'CAT'],
    ['speechiness_10cat', 'CAT'],
    ['acousticness_10cat', 'CAT'],
    ['instrumentalness_10cat', 'CAT'],
    ['liveness_10cat', 'CAT'],
    ['valence_10cat', 'CAT'], 
    ['tempo_10cat', 'CAT'],
    ['log10_popcat', 'CAT'], 
    ['appear_norm', 'FLT'], 
    ['appear_raw', 'FLT'],
    ['pop_by_pid', 'FLT'],
    ['80_20_LT', 'FLT'],
    ['genres_vec', 'VEC'],
    ['img_emb', 'VEC'],
    ['track_name_emb', 'VEC']
]

#MODEL
_C.MODEL = CfgNode()
_C.MODEL.ARCH = 'PINSAGE'

# pinsage
_C.MODEL.PINSAGE = CfgNode()

# pinsage params
_C.MODEL.PINSAGE.LAYERS = 2
_C.MODEL.PINSAGE.HIDDEN_SIZE = 128
_C.MODEL.PINSAGE.DROPOUT = 0.0 

# pinsage projection
_C.MODEL.PINSAGE.PROJECTION = CfgNode()
_C.MODEL.PINSAGE.PROJECTION.ALL_FEATURES = [] #List of all the features used in run 

#Hidden size per raw value embeddings 
_C.MODEL.PINSAGE.PROJECTION.EMB = [
    ['danceability', 8],
    ['energy', 8],
    ['loudness', 8],
    ['speechiness', 8],
    ['acousticness', 8],
    ['instrumentalness', 8],
    ['liveness', 8],
    ['valence', 8], 
    ['tempo', 8],
]
#List of raw value embeddings
_C.MODEL.PINSAGE.PROJECTION.CONCAT = [] 
    # 'liveness', 'instrumentalness',
    # 'speechiness', 'loudness', 'acousticness',  'genres']
_C.MODEL.PINSAGE.PROJECTION.ADD = [] #['tid', 'arid', 'alid']
_C.MODEL.PINSAGE.PROJECTION.NORMALIZE = False
_C.MODEL.PINSAGE.REPRESENTATION_NORMALIZE = False

_C.MODEL.PINSAGE.SCORER = 'DEFAULT'
_C.MODEL.PINSAGE.SCORER_BIAS = False


# _C.MODEL.PINSAGE.PROJECTION.CONCAT = [
#     'tempo', 'liveness', 'instrumentalness', 
#     'speechiness', 'loudness', 'acousticness', 'danceability', 'genre'
# ]

# _C.MODEL.PINSAGE.PROJECTION.ADD = ['tid', 'arid', 'alid']



_C.DATASET.SAMPLER = CfgNode()
_C.DATASET.SAMPLER.NODES_SAMPLER = CfgNode()
_C.DATASET.SAMPLER.NODES_SAMPLER.NAME = 'DEFAULT'
_C.DATASET.TRAIN_INDICES = 'train_indices'
_C.DATASET.SAMPLER.NODES_SAMPLER.PATH = ''
_C.DATASET.SAMPLER.NODES_SAMPLER.BATCH_SIZE = 32

_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER = CfgNode()
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.NAME = 'DEFAULT'
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER = CfgNode()
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER.RANDOM_WALK_LENGTH = 2
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER.RANDOM_WALK_RESTART_PROB = 0.5
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER.NUM_RANDOM_WALKS = 10
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER.NUM_NEIGHBORS = 3
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.DEFAULT_SAMPLER.NUM_LAYERS = 2
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.ADAPTIVE = False
_C.DATASET.SAMPLER.NEIGHBOR_SAMPLER.HOPS_AWAY = 3

_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.SOLVER = CfgNode()
_C.TRAIN.SOLVER.DECAY = True 
_C.TRAIN.SOLVER.GRAD_CLIPPING = False 
_C.TRAIN.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.TRAIN.LOSS = 'RAW_MARGIN_LOSS'
_C.TRAIN.UTILITY_EPOCHS = 10
_C.TRAIN.FAIR_EPOCHS = 1
_C.TRAIN.EPOCHS = 15
_C.TRAIN.BATCHES_PER_EPOCH = 50000  
_C.TRAIN.BATCHES_PER_UTILITY_EPOCH = 500
_C.TRAIN.BATCHES_PER_FAIRNESS_EPOCH = 500

_C.FAIR = CfgNode()
_C.FAIR.ALPHA = 1
_C.FAIR.FAIRNESS_BALANCE = 1  #gamma 
_C.FAIR.TOP_K = 10 
_C.FAIR.BOOST = 0.0
_C.FAIR.FEAT_SET = []
_C.FAIR.METHOD = "REDRESS"
_C.FAIR.NDCG_METHOD = "vanilla"
_C.FAIR.POP_FEAT = None  
_C.FAIR.IPW = False 

# Momentum.
_C.TRAIN.SOLVER.SGD = CfgNode()
_C.TRAIN.SOLVER.SGD.MOMENTUM = 0.9

# Momentum dampening.
_C.TRAIN.SOLVER.SGD.DAMPENING = 0.0

# Nesterov momentum.
_C.TRAIN.SOLVER.SGD.NESTEROV = True
_C.TRAIN.SOLVER.WEIGHT_DECAY = 0
_C.TRAIN.SOLVER.BASE_LR = 3e-5
_C.TRAIN.SOLVER.STEP_LRS = [
    [0, 1e-2],
    [1, 1e-3],
    [2, 1e-4],
    [3, 3e-5],
]
_C.FP16 = False
_C.OUTPUT_PATH = ''
_C.WANDB_NAME = ''

_C.REC = CfgNode()
_C.REC.K = 100 

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for my_project.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern recommended by the YACS repo.
    # It will be subsequently overwritten with local YAML.
    return _C.clone()

def load_config_from_file(path): 
    cfg = get_cfg_defaults()
    cfg.merge_from_file(path)
    return cfg 