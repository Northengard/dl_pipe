import os
import cv2
import json
from argparse import ArgumentParser
from imageio import get_reader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from face_recognition import face_locations


VID_NUM_FRAMES = 300
FPS = 30


def parse_args(argv):
    parser = ArgumentParser(description='train videos to image dataset')
    parser.add_argument('--vid-root', type=str, help='path to videos')
    parser.add_argument('--save-root', type=str, help='path to save images')
    parser.add_argument('-f', '--val-factor', type=float, help='validation size factor', default=0.2)
    parser.add_argument('-s', '--grab-step', type=int, help='grabbing images step', default=3)
    return parser.parse_args(argv)


def crop_faces(root_video_dir, videos_list, save_img_dir, meta_data, step=3):
    global VID_NUM_FRAMES
    print('saving into', save_img_dir)
    annotations = dict()
    labels = dict({'FAKE': 1, 'REAL': 0})
    for vid in tqdm(videos_list):
        v = get_reader(os.path.join(root_video_dir, vid))
        img_prefix = vid.split('.')[0]
        tq = tqdm(total=VID_NUM_FRAMES)
        for i, im in enumerate(v):
            if i % step == 0:
                faces = face_locations(im)
                for fc_coords in faces:
                    top, right, bottom, left = fc_coords
                    face = im[top:bottom, left:right]
                    face_path = img_prefix + "_" + str(i) + '.jpg'
                    annotations[face_path] = labels[meta_data[vid]['label']]
                    face_path = os.path.join(save_img_dir, face_path)
                    cv2.imwrite(face_path, face)
            tq.update(1)
    annot_path = os.path.join(save_img_dir, 'lables.json')
    with open(annot_path, 'w') as outpf:
        json.dump(annotations, outpf)


if __name__ == "__main__":
    args = parse_args(os.sys.argv[1:])

    grab_step = args.grab_step
    root_dir = args.vid_root
    root_img_dir = args.save_root

    print('videos from', root_dir)
    videos = os.listdir(root_dir)
    meta_index = list(map(lambda x: x[-4:] == 'json', videos)).index(True)
    meta = videos.pop(meta_index)
    meta = os.path.join(root_dir, meta)
    with open(meta, 'r') as metaf:
        meta = json.load(metaf)
    train, val = train_test_split(videos, test_size=args.val_factor, random_state=42)

    train_save_path = os.path.join(root_img_dir, 'train')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    crop_faces(root_video_dir=root_dir, videos_list=train,
               save_img_dir=train_save_path,
               meta_data=meta, step=grab_step)

    val_save_path = os.path.join(root_img_dir, 'validation')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    crop_faces(root_video_dir=root_dir, videos_list=val,
               save_img_dir=val_save_path,
               meta_data=meta, step=grab_step)
