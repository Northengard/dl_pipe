# os
import os
from argparse import ArgumentParser
from tqdm import tqdm

# framework
import torch

# custom
from data.datasets import get_dfc_video_dataset
from utils.storage import config_loader

from .train import model_init


def parse_args(arguments):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(arguments)


def test(model, data_loader, epoch, device, config):
    """
    model evaluation function
    :param model: model to eval
    :param data_loader: validation data loader
    :param epoch: int, last epoch number
    :param device: str, device to make computations
    :param config: dict, dictionary with train configurations
    :return: average_loss, score_value
    """
    model.eval()
    apply_sigmoid = config['test']['apply_sigmoid']
    batch_size = config['test']['batch_size']

    csv = None

    tq = tqdm(total=len(data_loader) * batch_size)
    tq.set_description('Test: Epoch {}'.format(epoch))
    with torch.no_grad():
        for itr, batch in enumerate(data_loader):
            images = batch['images']
            images = images.to(device)

            output = model(images)
            if apply_sigmoid:
                output = torch.sigmoid(output)

            tq.update(batch_size)
        tq.close()
    return csv


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, epoch = model_init(config, device)

    dataloader = get_dfc_video_dataset(config)
    test_res = test(model, epoch, dataloader, device, config)


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    main(cfg)
