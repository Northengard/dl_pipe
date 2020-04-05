import os
import yaml
import torch
import cv2
import json
from torch.utils.tensorboard import SummaryWriter


_SAVE_LOAD_PATH = 'snapshots'
_LOG_DIR = "logs"
_TBOARD_DEF_PATH = os.path.join(_LOG_DIR, 'training')


def _default_paths_init_():
    """
    initialize output paths such as logs and snapshots inner project directory.
    :return: None
    """
    global _SAVE_LOAD_PATH, _LOG_DIR, _TBOARD_DEF_PATH
    if not os.path.exists(_SAVE_LOAD_PATH):
        os.mkdir(_SAVE_LOAD_PATH)
    if not os.path.exists(_LOG_DIR):
        os.mkdir(_LOG_DIR)
    if not os.path.exists(_TBOARD_DEF_PATH):
        os.mkdir(_TBOARD_DEF_PATH)


_default_paths_init_()


def get_writer(config):
    """
    Get tensorboard writer according to given config
    :param config: dict
    :return: writer
    """
    global _TBOARD_DEF_PATH
    if config['experiment']['change_tensorboard_log_dir']:
        change_tboard_dir(new_path=config['experiment']['tensorboard_log_dir_path'])
    experiment_log_dir = os.path.join(_TBOARD_DEF_PATH, config['experiment']['name'])
    writer = SummaryWriter(experiment_log_dir)
    return writer


def change_tboard_dir(new_path):
    """
    replace default tensorboard log directory with desired
    :param new_path: new path to tensorboard logs
    :return: None
    """
    global _TBOARD_DEF_PATH
    if os.path.exists(new_path):
        _TBOARD_DEF_PATH = new_path


def config_loader(conf_path):
    """
    Load yaml config file
    :param conf_path: str, file path
    :return: yaml file content
    """
    with open(conf_path, 'r') as stream:
        yam = yaml.safe_load(stream)
    return yam


def load_weights(model, prefix, model_name, epoch):
    """
    Load network snapshot
    :param model: model, torch model to load snapshot
    :param prefix: str, snapshot prefix part
    :param model_name: str, model name (identifier)
    :param epoch: int, snapshot epoch
    :return: None
    """
    global _SAVE_LOAD_PATH
    file = os.path.join(_SAVE_LOAD_PATH,
                        '{}_{}_epoch_{}.pth'.format(prefix,
                                                    model_name,
                                                    epoch))
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])


def save_weights(model, prefix, model_name, epoch, parallel=True):
    """
    Save network snapshot
    :param model: model, torch model to save
    :param prefix: str, snapshot prefix part
    :param model_name: str, model name (identifier)
    :param epoch: int, snapshot epoch
    :param parallel: bool, set true if model learned on multiple gpu
    :return: None
    """
    global _SAVE_LOAD_PATH
    file = os.path.join(_SAVE_LOAD_PATH,
                        '{}_{}_epoch_{}.pth'.format(prefix,
                                                    model_name,
                                                    epoch))
    if torch.cuda.is_available() and parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'state_dict': state_dict}, file)


def load_image(imp_path):
    """
    Load image by given path using opencv-python (cv2)
    :param imp_path: str, path to image
    :return: image
    """
    return cv2.imread(imp_path)


def save_json(path, filename, data):
    """
    save the following json-serialisable data to json file
    :param path: str, directory to save file
    :param filename: str, desired filename must contain '.json' at the end
    :param data: data to save
    :return: None
    """
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as outpf:
        json.dump(data, outpf)
