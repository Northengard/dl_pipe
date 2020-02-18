import os
from argparse import ArgumentParser

import torch

from models.baseline import ForwardRegression
from utils.storage import config_loader


def parse_args(args):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(args)


def main(config):
    model = ForwardRegression(config)
    model = model.to('cuda')

    dummy = torch.rand(1, 3, 720, 1024)
    dummy = dummy.to('cuda')

    out = model(dummy)
    print('done')
    print(out.shape)


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    main(cfg)