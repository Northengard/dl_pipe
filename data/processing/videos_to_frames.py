# system
import os
from argparse import ArgumentParser
from tqdm import tqdm
import json

# multiprocessing
from multiprocessing import Process
import logging

# image/video handlers
import cv2
from imageio import get_reader

# face search
# from face_recognition import face_locations
from mtcnn.mtcnn import MTCNN

# util
import numpy as np
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.sys.path[0]))
    os.sys.path.append(root_dir)
from utils.logger import TqdmToLogger, get_logger

VID_NUM_FRAMES = 300
FPS = 30
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
LOG_DIR = list(map(lambda x: os.path.basename(x) == 'dl_pipe', os.sys.path)).index(True)
LOG_DIR = os.path.join(os.sys.path[LOG_DIR], 'logs', 'data_processing')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


def parse_args(argv):
    parser = ArgumentParser(description='train videos to image dataset')
    parser.add_argument('--vid-root',
                        type=str,
                        help='path to videos')
    parser.add_argument('--save-root',
                        type=str,
                        help='path to save images')
    parser.add_argument('-f', '--val-factor',
                        type=float,
                        default=0.2,
                        help='validation size factor')
    parser.add_argument('-s', '--grab-step',
                        type=int,
                        default=3,
                        help='grabbing images step')
    parser.add_argument('-n', '--n-jobs',
                        type=int,
                        default=12,
                        help='number of threads')
    parser.add_argument('-ff', '--face-bbox-expand-factor',
                        type=tuple,
                        default=(2, 2.5),
                        help='expand face bounding box width and height by factor parameters')
    return parser.parse_args(argv)


def _get_factorized_bbox(face_coords, bbox_factor):
    """
    scale face bounding box w, h by given factor
    :param face_coords: tuple, face coordinates
    :param bbox_factor: tuple, w and h scale factor
    :return: scaled_top, scaled_right, scaled_bottom, scaled_left
    """
    global IMAGE_WIDTH, IMAGE_HEIGHT
    left, top, face_w, face_h = face_coords
    new_w = face_w * bbox_factor[0]
    new_h = face_h * bbox_factor[1]
    bias_w = (new_w - face_w) // 2
    bias_h = (new_h - face_h) // 2
    # to avoid going abroad image
    top = int(max(0, top - bias_h))
    left = int(max(0, left - bias_w))
    bottom = int(min(IMAGE_HEIGHT, top + new_h))
    right = int(min(IMAGE_WIDTH, left + new_w))
    return top, right, bottom, left


def crop_faces(root_video_dir, videos_list, save_img_dir, meta_data, step, bbox_factor, name='0'):
    """
    Pass through the given videos and grubs faces from each frame with  the given step
    :param root_video_dir: directory contains video data
    :param videos_list: list of videos names to process
    :param save_img_dir: image saving directory
    :param meta_data: file with labels
    :param step: int, frames step to get faces,
    :param bbox_factor: face bbox scaling factor
    :param name: log file name
    :return: None
    """
    global VID_NUM_FRAMES
    log_path = name + '.log'
    logger = get_logger(log_path)
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    face_detector = MTCNN()

    logger.info(" ".join(['thread ', name, 'start']))
    logger.info('saving into ' + save_img_dir)

    annotations = dict()
    labels = dict({'FAKE': 1, 'REAL': 0})
    for vid in tqdm(videos_list, file=tqdm_out):
        v = get_reader(os.path.join(root_video_dir, vid))
        img_prefix = vid.split('.')[0]
        tq = tqdm(total=VID_NUM_FRAMES, file=tqdm_out)
        for frame_idx, image in enumerate(v):
            if frame_idx % step == 0:
                # faces = face_locations(image)
                faces = face_detector.detect_faces(image)
                for face_idx, fc_coords in enumerate(faces):
                    if fc_coords['confidence'] < 0.9:
                        continue
                    top, right, bottom, left = _get_factorized_bbox(fc_coords['box'], bbox_factor)
                    face = image[top:bottom, left:right]
                    face_path = "_".join([img_prefix, 'face_idx', str(face_idx), 'frame', str(frame_idx) + '.jpg'])
                    annotations[face_path] = labels[meta_data[vid]['label']]
                    face_path = os.path.join(save_img_dir, face_path)
                    cv2.imwrite(face_path, face)
                tq.update(step)
        tq.close()
    annot_path = os.path.join(save_img_dir, 'labels.json')
    with open(annot_path, 'w') as outpf:
        json.dump(annotations, outpf)


if __name__ == "__main__":
    args = parse_args(os.sys.argv[1:])

    n_jobs = args.n_jobs
    grab_step = args.grab_step
    root_dir = args.vid_root
    root_img_dir = args.save_root
    face_factor = args.face_bbox_expand_factor

    # parse video directory
    # get videos
    print('videos from', root_dir)
    videos = os.listdir(root_dir)

    # get metadata
    meta_index = list(map(lambda x: x[-4:] == 'json', videos)).index(True)
    meta = videos.pop(meta_index)
    meta = os.path.join(root_dir, meta)
    with open(meta, 'r') as metaf:
        meta = json.load(metaf)

    # train test split
    train, val = train_test_split(videos, test_size=args.val_factor, random_state=42)
    val_n_jobs = int(n_jobs * args.val_factor) + 1
    train_n_jobs = n_jobs - val_n_jobs

    # create train images subset
    train_save_path = os.path.join(root_img_dir, 'train')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    processes = list()
    train_split = np.array_split(np.array(train), train_n_jobs)
    print('spawning {} train handlers'.format(train_n_jobs))
    for i in range(train_n_jobs):
        process = Process(target=crop_faces, args=(root_dir, train_split[i], train_save_path, meta, grab_step,
                                                   face_factor, os.path.join(LOG_DIR, 'train_' + str(i))))
        processes.append(process)
        process.start()

    # create validation images subset
    val_save_path = os.path.join(root_img_dir, 'validation')
    if not os.path.exists(val_save_path):
        os.mkdir(val_save_path)
    val_split = np.array_split(np.array(val), val_n_jobs)
    print('spawning {} validation handlers'.format(val_n_jobs))
    for i in range(val_n_jobs):
        process = Process(target=crop_faces, args=(root_dir, val_split[i], val_save_path, meta, grab_step,
                                                   face_factor, os.path.join(LOG_DIR, 'val_' + str(i))))
        processes.append(process)
        process.start()

    for p in processes:
        ppid = p.pid
        print('wait for {} process'.format(ppid))
        p.join()
        print('process {} closed'.format(ppid))
