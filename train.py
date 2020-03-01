import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import models
from data.datasets import Dummy
from utils.storage import config_loader, load_weights, save_weights
from utils.handlers import AverageMeter, get_learning_rate
import losses


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
    start_epoch += 1

    if torch.cuda.is_available() and config['parallel']:
        model = DataParallel(model)

    return model, start_epoch


def metric_init(model_parameters, config):
    criterion = getattr(losses, config['loss'])()
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
    return criterion, optimizer, lr_scheduler, apply_sigmoid


def train(model, data_loader, criterion, optimizer, epoch, device, apply_sigmoid, config):
    model.train()

    loss_handler = AverageMeter()

    batch_size = config['batch_size']
    tq = tqdm(total=len(data_loader) * batch_size)
    tq.set_description('Train: Epoch {}, lr {:.2e}'.format(epoch + 1,
                                                           get_learning_rate(optimizer)))
    for itr, batch in enumerate(data_loader):
        images = batch['images']
        images = images.to(device)

        true_labels = batch['labels']
        true_labels = true_labels.to(device)

        output = model(images)
        if apply_sigmoid:
            output = torch.sigmoid(output)

        loss = criterion(output, true_labels)
        loss_handler.update(loss.item())
        loss.backward()

        if (itr + 1) % config['update_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        tq.update(batch_size)
        tq.set_postfix(loss='{:.4f}'.format(loss_handler.val),
                       avg_loss='{:.5f}'.format(loss_handler.avg))


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, start_epoch = model_init(config, device)

    criterion, optimizer, lr_scheduler, apply_sigmoid = metric_init(model.parameters(), config)

    data_loader = DataLoader(Dummy(len=100), batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'])

    for epoch in range(start_epoch, config['num_epochs']):
        train(model, data_loader, criterion, optimizer, epoch, device, apply_sigmoid, config)

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
