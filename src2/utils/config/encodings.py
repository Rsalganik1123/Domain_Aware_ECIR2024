from fvcore.common.config import CfgNode

_C = CfgNode()

#DATASET
_C.DATASET = CfgNode()
_C.DATASET.NAME = 'ENCODING'
_C.DATASET.DATA_PATH = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/final_pieces/complete_data_final_3way_with_emb_and_pos2.pkl'
_C.DATASET.USER_DF = 'df_playlist_info'
_C.DATASET.ITEM_DF = 'df_track'
_C.DATASET.INTERACTION_DF = 'df_playlist'
_C.DATASET.ITEM = 'track'
_C.DATASET.USER = 'playlist'
_C.DATASET.USER_ITEM_EDGE = 'contains'
_C.DATASET.ITEM_USER_EDGE = 'contained_by'

_C.DATASET.USER_ID = 'pid'
_C.DATASET.ITEM_ID = 'tid'
_C.DATASET.ITEM_FEATURES = [
    ['tid', 'CAT'],
    ['alid', 'CAT'],
    ['arid', 'CAT']
]

#MODEL
_C.MODEL = CfgNode()
_C.MODEL.ARCH = 'PINSAGE'

# pinsage
_C.MODEL.PINSAGE = CfgNode()

# pinsage params
_C.MODEL.PINSAGE.LAYERS = 2
_C.MODEL.PINSAGE.HIDDEN_SIZE = 128

# pinsage projection
_C.MODEL.PINSAGE.PROJECTION = CfgNode()
_C.MODEL.PINSAGE.PROJECTION.FEATURES = []
#                                     #'tid', 'alid', 'arid',
#                                     'tempo', 'liveness', 'instrumentalness',
#                                     'danceability', 'energy', 'valence'
#                                     'speechiness', 'loudness', 'acousticness'] 
#                                     #'img_emb', 'track_name_emb', 'genres' ]

_C.MODEL.PINSAGE.PROJECTION.EMB = [
    ['tid', 2258906],
    ['alid', 295859],
    ['arid', 734683],
    ['danceability', 8],
    ['energy', 8],
    ['loudness', 8],
    ['speechiness', 8],
    ['acousticness', 8],
    ['instrumentalness', 8],
    ['liveness', 8],
    ['valence', 8], 
    ['tempo', 8],
    ['popularity_10cat', 16],
    ['followers_10cat', 16],
]
_C.MODEL.PINSAGE.PROJECTION.CONCAT = [
    'liveness', 'instrumentalness',
    'speechiness', 'loudness', 'acousticness',  'genres']
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

_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.SOLVER = CfgNode()
_C.TRAIN.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.TRAIN.LOSS = 'RAW_MARGIN_LOSS'
_C.TRAIN.UTILITY_EPOCHS = 10
_C.TRAIN.FAIR_EPOCHS = 10
_C.TRAIN.EPOCHS = 15
_C.TRAIN.BATCHES_PER_EPOCH = 50000  
_C.TRAIN.BATCHES_PER_UTILITY_EPOCH = 50000
_C.TRAIN.BATCHES_PER_FAIRNESS_EPOCH = 50000

_C.FAIR = CfgNode()
_C.FAIR.ALPHA = 1
_C.FAIR.FAIRNESS_BALANCE = 1 
_C.FAIR.TOP_K = 10 

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