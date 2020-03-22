# os
import os
import sys
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

# framework
import torch
from numpy import inf
from torch import optim
from torch.nn import DataParallel

# custom
import models
import losses
import metrics
from data.datasets import get_dfc_dataset
from utils.handlers import AverageMeter, get_learning_rate
from utils.storage import config_loader, load_weights, save_weights, get_writer
from utils.logger import Logger


def parse_args(arguments):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(arguments)


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
    metric = getattr(metrics, config['metric'])
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


def validation(model, data_loader, criterion, metric, epoch, device, apply_sigmoid, config):
    """
    model evaluation function
    :param model: model to eval
    :param data_loader: validation data loader
    :param criterion: loss function
    :param metric: score function
    :param epoch: int, last epoch number
    :param device: str, device to make computations
    :param apply_sigmoid: bool, utils flag. Please, see metric_init for details.
    :param config: dict, dictionary with train configurations
    :return: average_loss, score_value
    """
    model.eval()

    loss_handler = AverageMeter()
    metric_handler = AverageMeter()

    batch_size = config['validation']['batch_size']
    tq = tqdm(total=len(data_loader) * batch_size)
    tq.set_description('Val: Epoch {}'.format(epoch))

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

        score = metric(output, true_labels)
        metric_handler.update(score)

        tq.update(batch_size)
        tq.set_postfix(loss='{:.4f}'.format(loss_handler.val),
                       avg_loss='{:.5f}'.format(loss_handler.avg),
                       avg_score='{:.5f}'.format(metric_handler.avg))

        return loss_handler.avg, metric_handler.avg


def train(model, data_loader, criterion, optimizer, epoch, device, apply_sigmoid, config, summary_writer):
    """
    Model train function
    :param model: model to train
    :param data_loader: train data loader
    :param criterion: loss function
    :param optimizer: score function
    :param epoch: int, last epoch number
    :param device: str, device to make computations
    :param apply_sigmoid: bool, utils flag. Please, see metric_init for details.
    :param config: dict, dictionary with train configurations
    :param summary_writer: tensorboard writer to flush train artifacts
    """
    model.train()

    loss_handler = AverageMeter()

    batch_size = config['train']['batch_size']
    train_len = len(data_loader)

    tq = tqdm(total=train_len * batch_size)
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

        summary_writer.add_scalar('Train/' + config['loss'], loss, (itr + 1) + train_len * epoch)

        tq.update(batch_size)
        tq.set_postfix(loss='{:.4f}'.format(loss_handler.val),
                       avg_loss='{:.5f}'.format(loss_handler.avg))


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, start_epoch = model_init(config, device)

    criterion, metric, optimizer, lr_scheduler, apply_sigmoid = metric_init(model.parameters(), config)

    train_loader, val_loader = get_dfc_dataset(config)
    train_loader_len = len(train_loader)

    writer = get_writer(config)

    dummy = torch.rand(1, 3, *config['input_size'])
    dummy = dummy.to(device)
    writer.add_graph(model, input_to_model=dummy)

    best_loss = inf
    best_score = 0
    for epoch in range(start_epoch, config['num_epochs']):
        train(model=model, data_loader=train_loader,
              criterion=criterion, optimizer=optimizer,
              epoch=epoch, device=device,
              apply_sigmoid=apply_sigmoid,
              config=config, summary_writer=writer)

        loss, score = validation(model=model, data_loader=val_loader,
                                 criterion=criterion, metric=metric,
                                 device=device, apply_sigmoid=apply_sigmoid,
                                 epoch=epoch, config=config)

        writer.add_scalar('Train/learning_rate', get_learning_rate(optimizer), train_loader_len * epoch)
        writer.add_scalar('Validation/' + config['loss'], loss, (epoch + 1) * train_loader_len)
        writer.add_scalar('Validation/' + config['loss'], loss, (epoch + 1) * train_loader_len)

        lr_scheduler.step(epoch)
        if ((epoch + 1) % config['save_freq']) == 0:
            save_weights(model, config['prefix'], config['model']['name'], (epoch + 1), config['parallel'])
        if best_loss > loss:
            best_loss = loss
            best_epoch = epoch + 1
            if best_score < score:
                best_score = score
            save_weights(model, config['prefix'], 'model', 'best_loss-' + str(best_epoch), config['parallel'])
        elif best_score < score:
            best_score = score
            best_epoch = epoch + 1
            save_weights(model, config['prefix'], 'model', 'best_score-' + str(best_epoch), config['parallel'])


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    date = str(datetime.now())
    sys.stdout = Logger(exp_name=cfg['experiment']['name'], filename=f"RUN_{date}.log", stderr=True)
    sys.stderr = Logger(exp_name=cfg['experiment']['name'], filename=f"RUN_{date}.err", stderr=False)
    print('Run Logging of execution')
    main(cfg)
