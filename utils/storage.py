import os
import yaml
import torch
import cv2
import logging
from io import StringIO
from torch.utils.tensorboard import SummaryWriter


_save_load_path = 'snapshots'
_tensorboard_default_path = os.path.join('logs', 'training')


def _default_paths_init_():
    global _save_load_path, _tensorboard_default_path
    if not os.path.exists(_save_load_path):
        os.mkdir(_save_load_path)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(_tensorboard_default_path):
        os.mkdir(_tensorboard_default_path)


_default_paths_init_()


def get_writer(config):
    global _tensorboard_default_path
    if config['experiment']['change_tensorboard_log_dir']:
        change_tboard_dir(new_path=config['experiment']['tensorboard_log_dir_path'])
    experiment_log_dir = os.path.join(_tensorboard_default_path, config['experiment']['name'])
    writer = SummaryWriter(experiment_log_dir)
    return writer


def change_tboard_dir(new_path):
    global _tensorboard_default_path
    if os.path.exists(new_path):
        _tensorboard_default_path = new_path


def config_loader(conf_path):
    with open(conf_path, 'r') as stream:
        yam = yaml.safe_load(stream)
    return yam


def load_weights(model, prefix, model_type, epoch):
    global _save_load_path
    file = os.path.join(_save_load_path,
                        '{}_{}_epoch_{}.pth'.format(prefix,
                                                    model_type,
                                                    epoch))
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])


def save_weights(model, prefix, model_type, epoch, parallel=True):
    global _save_load_path
    file = os.path.join(_save_load_path,
                        '{}_{}_epoch_{}.pth'.format(prefix,
                                                    model_type,
                                                    epoch))
    if torch.cuda.is_available() and parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'state_dict': state_dict}, file)


def load_image(imp_path):
    return cv2.imread(imp_path)


class TqdmToLogger(StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def get_logger(logger_path):
    # create logger for prd_ci
    logger_name = os.path.basename(logger_path)
    log = logging.getLogger(logger_name)
    log.setLevel(level=logging.DEBUG)

    # create file handler for logger.
    fh = logging.FileHandler(logger_path)
    fh.setLevel(level=logging.DEBUG)

    log.addHandler(fh)
    return log
