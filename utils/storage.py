import os
import yaml
import torch


_save_load_path = 'snapshots'


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
