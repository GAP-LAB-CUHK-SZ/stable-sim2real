from sampler.registry import Registry, build_from_config

def build_func(cfg, registry, **kwargs):
    """
    Except for config, if passing a list of dataset config, then return the concat type of it
    """
    return build_from_config(cfg, registry, **kwargs)

SAMPLER = Registry("SAMPLER", build_func=build_func)
