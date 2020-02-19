import os
from argparse import ArgumentParser

import torch
from torch.nn import DataParallel
from torch import optim

import models
from utils.storage import config_loader, load_weights, save_weights


def parse_args(arguments):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(arguments)


def model_init(config, device):
    model = getattr(models, config['model']['name'])(config['model'])
    model = model.to(device)

    # load model
    start_epoch = 0
    if config['snapshot']['use']:
        load_weights(model, config['prefix'], 'model', config['snapshot']['epoch'])
        if type(config['snapshot']['epoch']) == int:
            start_epoch = config['snapshot']['epoch']
        else:
            start_epoch = int(config['snapshot']['epoch'].split('-')[-1])

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    return model, start_epoch


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, start_epoch = model_init(config, device)

    dummy = torch.rand(1, 3, 720, 1024)
    dummy = dummy.to(device)

    out = model(dummy)
    print('done')
    print(out.shape)
    save_weights(model, config['prefix'], config['model']['name'], 0, config['parallel'])
    print('saved')


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    main(cfg)
