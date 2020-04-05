# os
import os
from argparse import ArgumentParser
from tqdm import tqdm

# framework
import numpy as np
import torch
from csv import writer as csv_writer

# custom
from data.datasets import get_dfc_video_dataset
from utils.storage import config_loader
from utils import model_init
from utils.handlers import AverageMeter


def parse_args(arguments):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(arguments)


def test(model, data_set, epoch, device, config):
    """
    model evaluation function
    :param model: model to eval
    :param data_set: validation data loader
    :param epoch: int, last epoch number
    :param device: str, device to make computations
    :param config: dict, dictionary with train configurations
    :return: average_loss, score_value
    """
    model.eval()
    apply_sigmoid = config['test']['apply_sigmoid']

    with open('submition.csv', 'w', newline='\n') as csvfile:
        csv = csv_writer(csvfile, delimiter=',')

        tq = tqdm(total=len(data_set))
        tq.set_description('Test: Epoch {}'.format(epoch))
        stat = AverageMeter()
        with torch.no_grad():
            for vid_idx in range(len(data_set)):
                stat.reset()
                reader = data_set[vid_idx]
                indexes = np.random.choice(list(range(300)), config['test']['num_samples'])
                for idx in indexes:
                    frame = reader.get_data(idx)
                    image = data_set.apply_transform(frame)
                    image = image.to(device)

                    output = model(image)
                    if apply_sigmoid:
                        output = torch.sigmoid(output)
                    output = output.cpu().numpy()
                    output = np.argmax(output)
                    stat.update(output)
                output = 1 if stat.avg > 0.5 else 0
                csv.writerow([data_set.get_last_vid_name(), output])
                tq.update()
            tq.close()


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, epoch = model_init(config, device)
    data_set = get_dfc_video_dataset(config)
    test(model, data_set, epoch, device, config)


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    main(cfg)
