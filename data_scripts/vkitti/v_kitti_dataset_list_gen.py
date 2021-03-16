import random
from argparse import ArgumentParser
from os import listdir
from os.path import join, splitext

import cv2
import numpy as np

from config.config import Config, DIR_V_KITTI

cfg = Config.get_config()

SCENE_FOLDERS = [f'Scene{i:02}' for i in [1, 2, 6, 18, 20]]
VARIATIONS = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset',
              '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']
CAMERAS = ['Camera_0', 'Camera_1']


def _generate_train_slices_from_val_slices(val_slices, max_index_exclusive):
    """from list of val_slices generates list of train_slices"""
    val_slices.sort()
    train_slices = []
    train_start = 0

    val_slices.append((max_index_exclusive, max_index_exclusive))  # guard
    for val_start, val_end in val_slices:
        train_end = val_start - 1
        if train_start <= train_end:
            train_slices.append((train_start, train_end))
        train_start = val_end + 1
    val_slices.pop()  # remove guard

    return train_slices


def _generate_val_slices(nr_items_total, nr_items_val, min_len_val_slice=10, max_len_val_slice=30,
                         min_items_between_slices=30):
    """
    generates intervals of frames for val set

    ex: nr_items_total is 1000, return [(2,20), (600,630)] frames between 2 and 20, 600 and 630 are for validation
    """
    nr_items = nr_items_total
    used = [False] * nr_items
    val_slices = []

    while nr_items_val > 0:
        start = random.randint(0, nr_items - 1)
        if used[start]:
            continue

        left = used[start - 1::-1]
        right = used[start + 1:]

        left_first = start - 1 - left.index(True) if True in left else -1
        right_first = start + 1 + right.index(True) if True in right else nr_items

        if start - left_first < min_items_between_slices:
            continue

        max_poss_len = min(right_first - start, max_len_val_slice, nr_items_val)
        if nr_items_val >= min_len_val_slice and max_poss_len < min_len_val_slice:
            continue

        curr_len = random.randint(min_len_val_slice,
                                  max_poss_len) if min_len_val_slice <= max_poss_len else max_poss_len  # few remaining elements

        if right_first - (start + curr_len) < min_items_between_slices:
            continue

        used[start:(start + curr_len)] = [True] * curr_len
        nr_items_val = nr_items_val - curr_len
        val_slices.append((start, start + curr_len - 1))

    val_slices.sort()
    return val_slices


def _write_val_slices_to_file(file, scenes_dict):
    for scene, slices in sorted(scenes_dict.items()):
        file.write(scene + '\n')
        for slic in slices:
            file.write(f'{slic[0]}, {slic[1]}\n')
        file.write('\n')


def _read_val_slices_from_file(file_name):
    slices = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()

        curr_scene = ''
        new_scene = True
        for line in lines:
            line = line.strip()
            if line.startswith('#'):  # pass over comments
                continue

            if not line:
                new_scene = True
                continue

            if new_scene:
                curr_scene = line
                slices[curr_scene] = []
                new_scene = False
            else:
                slices[curr_scene].append(tuple(map(int, line.split(','))))

    return slices


def _get_dir_size_and_extensions(dir_path):
    filenames = listdir(dir_path)
    return len(filenames), splitext(filenames[0])[1]


def _write_dataset_list(file, data_dir, data_ext, label_dir, label_ext, slices):
    for slic in slices:
        for i in range(slic[0], slic[1] + 1):
            line = f'{data_dir}/rgb_{i:05}{data_ext} {label_dir}/classgt_{i:05}{label_ext}\n'
            file.write(line)


def make_val_slices_file(args, train_split=0.8):
    """writes the json file with val intervals"""
    slices = {}
    for scene in args.scene:
        dir = join(args.data_root, scene, 'clone/frames/rgb/Camera_0')
        files = listdir(dir)
        frames_val = int(len(files) * (1 - train_split))
        slices[scene] = _generate_val_slices(len(files), frames_val, 10, 30, 30)

    savefile = join(args.save_dir, args.slices_filename)
    with open(savefile, 'w') as file:
        file.write(f'#train: {train_split}, val:{1 - train_split}\n')
        _write_val_slices_to_file(file, slices)


def load_val_slices_file(args):
    filename = join(args.save_dir, args.slices_filename)
    slices = _read_val_slices_from_file(filename)
    return slices


def _get_datalist_filename(args, type):
    dest_name, dest_ext = splitext(args.datalist_filename)
    variations = 'all' if args.variation == VARIATIONS else '_'.join(args.variation)
    cameras = 'c' + ''.join([s.split('_')[1] for s in args.camera])
    return f'{dest_name}_{variations}_{cameras}_{type}{dest_ext}'


def gen_segmentation_list(args):
    filename = join(args.save_dir, args.datalist_filename)

    val_slices = load_val_slices_file(args)

    file_header = f'#scenes: {"all" if args.scene == SCENE_FOLDERS else args.scene}, \
variations: {"all" if args.variation == VARIATIONS else args.variation}, \
cameras: {"all" if args.camera == CAMERAS else args.camera}, \
file: {args.slices_filename}\n'

    dest_name, dest_ext = splitext(args.datalist_filename)

    with open(join(args.save_dir, _get_datalist_filename(args, 'train')), 'w') as file_train, \
            open(join(args.save_dir, _get_datalist_filename(args, 'val')), 'w') as file_val:
        file_train.write(file_header)
        file_val.write(file_header)
        for scene in args.scene:
            for variation in args.variation:
                for camera in args.camera:
                    path_data = join(scene, variation, 'frames', 'rgb', camera)
                    path_label = join(scene, variation, 'frames', 'classSegmentation', camera)

                    data_size, data_ext = _get_dir_size_and_extensions(join(args.data_root, path_data))
                    _, label_ext = _get_dir_size_and_extensions(join(args.data_root, path_label))

                    val_slice = val_slices[scene]
                    train_slice = _generate_train_slices_from_val_slices(val_slice, data_size)

                    _write_dataset_list(file_train, path_data, data_ext, path_label, label_ext, train_slice)
                    _write_dataset_list(file_val, path_data, data_ext, path_label, label_ext, val_slice)


def _read_data_list(list_filename, type):
    with open(list_filename, 'r') as file:
        lines = [line.split() for line in file.readlines() if not line.strip().startswith('#')]
        return [(data, label, type) for data, label in lines]


def make_video(args, train_filename=None, val_filename=None):
    train_filename = '../../resources/dataset_lists/v_kitti_seg_clone_c0_train.txt'
    val_filename = '../../resources/dataset_lists/v_kitti_seg_clone_c0_val.txt'

    full_list = _read_data_list(val_filename, 'val')
    full_list.extend(_read_data_list(train_filename, 'train'))
    full_list.sort()

    full_list = [(cv2.imread(join(args.data_root, data)), cv2.imread(join(args.data_root, label.replace('trainId_', ''))), type)
                 for data, label, type in full_list]

    def transform(p):
        if p[2] == 'val':
            cv2.putText(p[0], 'VALIDATION', (0, 50), cv2.QT_FONT_NORMAL, 1, (255, 0, 0), 2)
            cv2.putText(p[1], 'VALIDATION', (0, 50), cv2.QT_FONT_NORMAL, 1, (255, 0, 0), 2)
        return p[0], p[1]

    full_list = map(transform, full_list)
    conc_list = [np.concatenate((data, label)) for data, label in full_list]

    h, w, _ = conc_list[0].shape
    vid_size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    vid = cv2.VideoWriter(join(args.save_dir, 'movie.avi'), fourcc, 4, vid_size)
    for img in conc_list:
        vid.write(img)
    vid.release()


def main(args):
    # gen_segmentation_list(args)
    make_video(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-root', default=DIR_V_KITTI)
    parser.add_argument('--scene', nargs='+', choices=SCENE_FOLDERS + ['all'], default=['all'])
    parser.add_argument('--variation', nargs='+', choices=VARIATIONS + ['all'], default=['clone'])
    parser.add_argument('--camera', nargs='+', choices=CAMERAS + ['all'], default=['Camera_0'])

    parser.add_argument('--save_dir', default='../dataset_lists/')
    parser.add_argument('--slices-filename', default='v_kitti_val_slices_80_20.txt')
    parser.add_argument('--datalist-filename', default='v_kitti_seg.txt')

    args = parser.parse_args()
    if 'all' in args.scene:
        args.scene = SCENE_FOLDERS
    if 'all' in args.variation:
        args.variation = VARIATIONS
    if 'all' in args.camera:
        args.camera = CAMERAS

    main(args)
