import os
import yaml


def config_loader(conf_path):
    with open(conf_path, 'r') as stream:
        yam = yaml.safe_load(stream)
    return yam