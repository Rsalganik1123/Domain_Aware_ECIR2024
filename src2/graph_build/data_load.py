from src2.utils.registry import Registry

DATASET_REGISTRY = Registry()

def build_dataset(cfg):
    name = cfg.DATASET.NAME
    return DATASET_REGISTRY.get(name)(cfg)