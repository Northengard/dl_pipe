# os
import os
import sys
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

# framework
import torch
from numpy import inf

# custom
from data import datasets
from utils.handlers import AverageMeter, get_learning_rate
from utils import model_init, metric_init
from utils.storage import config_loader, save_weights, get_writer
from utils.logger import Logger


def parse_args(arguments):
    parser = ArgumentParser(description='network training script')
    parser.add_argument('-c', '--config', type=str, help='config path',
                        default='./configs/ForwardRegression_train.yaml')
    return parser.parse_args(arguments)


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
    metric.reset()

    loss_handler = AverageMeter()

    batch_size = config['validation']['batch_size']
    tq = tqdm(total=len(data_loader) * batch_size)
    tq.set_description('Val: Epoch {}'.format(epoch))
    with torch.no_grad():
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

            tq.update(batch_size)
            tq.set_postfix(loss='{:.4f}'.format(loss_handler.val),
                           avg_loss='{:.5f}'.format(loss_handler.avg),
                           avg_score='{:.5f}'.format(score))
        tq.close()
        return loss_handler.avg, score


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
    tq.close()


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, start_epoch = model_init(config, device)

    criterion, metric, optimizer, lr_scheduler, apply_sigmoid = metric_init(model.parameters(), config)

    train_loader = getattr(datasets, config['dataset_name'])(config, is_train=True)
    val_loader = getattr(datasets, config['dataset_name'])(config, is_train=False)
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
        writer.add_scalar('Validation/' + config['metric'], score, (epoch + 1) * train_loader_len)

        lr_scheduler.step(epoch)
        if ((epoch + 1) % config['save_freq']) == 0:
            save_weights(model, config['prefix'], config['model']['name'], (epoch + 1), config['parallel'])
        if best_loss > loss:
            best_loss = loss
            best_epoch = epoch + 1
            if best_score < score:
                best_score = score
            save_weights(model, config['prefix'], config['model']['name'], 'best_loss-' + str(best_epoch),
                         config['parallel'])
        elif best_score < score:
            best_score = score
            best_epoch = epoch + 1
            save_weights(model, config['prefix'], config['model']['name'], 'best_score-' + str(best_epoch),
                         config['parallel'])


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])
    cfg = config_loader(args.config)
    date = str(datetime.now())
    sys.stdout = Logger(exp_name=cfg['experiment']['name'], filename=f"RUN_{date}.log", stderr=True)
    sys.stderr = Logger(exp_name=cfg['experiment']['name'], filename=f"RUN_{date}.err", stderr=False)
    print('Run Logging of execution')
    main(cfg)
