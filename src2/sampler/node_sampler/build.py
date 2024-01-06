from src2.utils.registry import Registry

NODE_SAMPLER_REGISTRY = Registry()


def build_nodes_sampler(g, cfg):
    name = cfg.DATASET.SAMPLER.NODES_SAMPLER.NAME
    return NODE_SAMPLER_REGISTRY.get(name)(g, cfg)

