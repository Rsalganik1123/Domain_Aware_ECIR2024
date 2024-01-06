import torch
import torch.nn.functional as F


def build_optimizer(model, cfg):

    parameters = model.parameters()
    opt_method = cfg.TRAIN.SOLVER.OPTIMIZING_METHOD
    base_lr = cfg.TRAIN.SOLVER.BASE_LR
    if opt_method == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=base_lr)
    else:
        sgd_cfg = cfg.TRAIN.SOLVER.SGD
        momentum = sgd_cfg.MOMENTUM
        nesterov = sgd_cfg.NESTEROV
        dampening = sgd_cfg.DAMPENING
        weigth_decay = cfg.TRAIN.SOLVER.WEIGHT_DECAY
        optimizer = torch.optim.SGD(parameters, lr=base_lr, momentum=momentum, nesterov=nesterov,
                                    dampening=dampening, weight_decay=weigth_decay)
    return optimizer
