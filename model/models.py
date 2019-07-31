def create_model(cfg, writer=None):

    model_name = cfg.MODEL
    print(model_name)
    if model_name == 'trecg':
        from .trecg_model import TRecgNet
        model = TRecgNet(cfg, writer)
    elif model_name == 'fusion':
        from .fusion import Fusion
        model = Fusion(cfg, writer)
    else:
        raise ValueError('Model {0} not recognized'.format(model_name))
    return model