from fvcore.common.config import CfgNode

_C = CfgNode()

_C.MODEL = CfgNode()
_C.MODEL.ARCH = 'PINSAGE'

# pinsage
_C.MODEL.PINSAGE = CfgNode()

# pinsage params
_C.MODEL.PINSAGE.LAYERS = 2
_C.MODEL.PINSAGE.HIDDEN_SIZE = 128

# pinsage projection
_C.MODEL.PINSAGE.PROJECTION = CfgNode()
_C.MODEL.PINSAGE.PROJECTION.FEATURES = ['key', 'tempo_5cat', 'livness_5cat', 'instrumentalness_3cat',
                                        'speechiness_10cat', 'loudness_10cat', 'acousticness_10cat',
                                        'artist_id', 'album_id', 'music_continous_features',
                                        'genre_old_vec', 'genres_vec', 'tid']

_C.MODEL.PINSAGE.PROJECTION.EMB = [
    ['tid', 128],
    ['album_id', 128],
    ['artist_id', 128],
    ['key', 16],
    ['tempo_5cat', 8],
    ['livness_5cat', 8],
    ['instrumentalness_3cat', 4],
    ['speechiness_10cat', 16],
    ['loudness_10cat', 16],
    ['acousticness_10cat', 16]
]
_C.MODEL.PINSAGE.PROJECTION.CONCAT = [
    'key', 'tempo_5cat', 'livness_5cat',
    'instrumentalness_3cat', 'speechiness_10cat', 'loudness_10cat',
    'acousticness_10cat', 'music_continous_features', 'genre_old_vec', 'genres_vec'
]
_C.MODEL.PINSAGE.PROJECTION.ADD = ['tid', 'artist_id', 'album_id']
_C.MODEL.PINSAGE.PROJECTION.NORMALIZE = False
_C.MODEL.PINSAGE.REPRESENTATION_NORMALIZE = False

_C.MODEL.PINSAGE.SCORER = 'DEFAULT'
_C.MODEL.PINSAGE.SCORER_BIAS = False

_C.DATASET = CfgNode()
_C.DATASET.NAME = 'SPOTIFY_MUSIC'
_C.DATASET.DATA_PATH = '/home/NOBACKUP/mzhou3/599/ns_music_all_data.p'
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
    ['key', 'CAT'],
    ['tempo_5cat', 'CAT'],
    ['livness_5cat', 'CAT'],
    ['instrumentalness_3cat', 'CAT'],
    ['speechiness_10cat', 'CAT'],
    ['loudness_10cat', 'CAT'],
    ['acousticness_10cat', 'CAT'],
    ['artist_id', 'CAT'],
    ['album_id', 'CAT'],
    ['music_continous_features', 'VEC'],
    ['genre_old_vec', 'VEC'],
    ['genres_vec', 'VEC']
]

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
_C.TRAIN.EPOCHS = 10
_C.TRAIN.BATCHES_PER_EPOCH = 50000

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