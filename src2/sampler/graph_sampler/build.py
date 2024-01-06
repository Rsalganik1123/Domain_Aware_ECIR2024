from src2.utils.registry import Registry

GRAPH_SAMPLER_REGISTRY = Registry()


def build_graph_sampler(g, cfg):
    name = cfg.DATASET.SAMPLER.NEIGHBOR_SAMPLER.NAME
    return GRAPH_SAMPLER_REGISTRY.get(name)(g, cfg)