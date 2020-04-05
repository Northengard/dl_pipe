import torch
from torch import optim
from torch.nn import DataParallel

import losses
import metrics

import models
from utils.storage import load_weights


def model_init(config, device):
    """
    Initialize model according the given config
    :param config: dict, object with train configurations
    :param device: device to store model
    :return: model and start epoch to restore learning order
    """
    model = getattr(models, config['model']['name'])(config['model'])
    model = model.to(device)

    # load model
    start_epoch = 0
    if config['snapshot']['use']:
        load_weights(model, config['prefix'], config['model']['name'], config['snapshot']['epoch'])
        if type(config['snapshot']['epoch']) == int:
            start_epoch = config['snapshot']['epoch']
        else:
            start_epoch = int(config['snapshot']['epoch'].split('-')[-1])
        start_epoch += 1

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    return model, start_epoch


def metric_init(model_parameters, config):
    """
    Initialize loss, metric, scheduler and optimizer according the given config \n
    Output Description: \n
    criterion - loss function \n
    metric - score function used for model evaluation \n
    optimizer - model optimizer. Such as Adam or SGD \n
    lr_scheduler - scheduler to deal with optimizer managing \n
    apply_sigmoid - boolean flag to applying sigmoid function over model output if it's needed. Depends on loss. \n

    :param model_parameters: model.parameters() as input is required
    :param config: dict, object with train configurations
    :return: criterion, metric, optimizer, lr_scheduler, apply_sigmoid
    """
    criterion = getattr(losses, config['loss'])()
    metric = getattr(metrics, config['metric'])()
    optimizer = getattr(optim, config['optimizer'])(model_parameters,
                                                    lr=config['learning_rate']['value'],
                                                    weight_decay=config['weight_decay'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=config['learning_rate']['decay'],
                                                        patience=config['learning_rate']['no_improve'],
                                                        min_lr=config['learning_rate']['min_val'])
    # TODO: optional sheduler type
    if config['loss'] == 'bce':
        apply_sigmoid = True
    else:
        apply_sigmoid = False
    return criterion, metric, optimizer, lr_scheduler, apply_sigmoid
