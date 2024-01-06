from src2.utils.registry import Registry

MODEL_REGISTRY = Registry()

def build_model(full_graph, cfg):
    model = MODEL_REGISTRY[cfg.MODEL.ARCH](full_graph, cfg)
    return model