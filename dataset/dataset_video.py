import json
import os
import random
from collections import OrderedDict, namedtuple
from os.path import exists, join, expanduser

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config.config import Config
from dataset.dataset_transforms import transform_v_kitti_png_flow


class CityscapesVideo(Dataset):
    def __init__(self, root, subset='train', pair_transform=None, input_transform=None, target_transform=None, nr_frames: int = 1, return_filename: bool = True, flow_dir: str = None):
        """ pair_transform and input_transform should work on list of images
            returns RGB list of input images
        """

        self.nr_frames = nr_frames
        self.images_root = os.path.join(root, 'leftImg8bit_sequence/')
        self.labels_root = os.path.join(root, 'gtFine/')
        self.flow_root = os.path.join(root, flow_dir) if flow_dir else None

        assert exists(self.images_root), f'{self.images_root} not existing'
        assert exists(self.labels_root), f'{self.labels_root} not existing'
        assert self.flow_root is None or exists(self.flow_root), f'{self.flow_root} not existing'

        self.images_root += subset
        self.labels_root += subset
        if self.flow_root:
            self.flow_root = join(self.flow_root, subset)

        self.filenames_labels = [join(root, f) for root, _, files in os.walk(expanduser(self.labels_root)) for f in
                                 files if f.endswith("_labelTrainIds.png")]
        self.filenames_labels.sort()

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.return_filename = return_filename

    def __getitem__(self, index):
        filename_label = self.filenames_labels[index]

        parts = filename_label.replace('\\', '/').split('/')[-1].split('_')
        city, seq, frame = parts[0], parts[1], parts[2]

        label_image = Image.open(filename_label).convert('L')
        input_images = []
        for idx in range(-self.nr_frames + 1, 1):
            x = int(frame) + idx
            filename = join(self.images_root, city, f'{city}_{seq}_{x:06}_leftImg8bit.png')
            try:
                img = Image.open(filename).convert('RGB')
            except:
                print(f'problem with file {filename}')
                raise RuntimeError
            input_images.append(img)

        flow_images = []
        if self.flow_root:
            for idx in range(-self.nr_frames + 2, 1):
                x = int(frame) + idx
                filename = join(self.flow_root, city, f'{city}_{seq}_{x:06}_backFlow.tpf')
                flow = torch.load(filename)
                flow_images.append(flow)

        if self.input_transform:
            input_images = self.input_transform(input_images)
        if self.target_transform:
            label_image = self.target_transform(label_image)
        if self.pair_transform:
            input_images, label_image = self.pair_transform(input_images, [label_image])

        if self.return_filename:
            return input_images, label_image, flow_images, True, filename_label
        else:
            return input_images, label_image, flow_images, True

    def shuffle(self):
        self.filenames_labels = random.sample(self.filenames_labels, len(self.filenames_labels))

    def __len__(self):
        return len(self.filenames_labels)


class CityscapesVideoSmall(CityscapesVideo):

    def __init__(self, root, subset='train', pair_transform=None, input_transform=None, target_transform=None, nr_frames: int = 1, return_filename: bool = True, flow_dir: str = None):
        super().__init__(root, subset, pair_transform, input_transform, target_transform, nr_frames, return_filename, flow_dir)
        self.filenames_labels = self.filenames_labels[:5]


def read_vkitti_png_flow(flow_fn):
    """Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"""
    # https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0  # or another value (e.g., np.nan)
    return out_flow


def read_v_kitti_sequences(filename: str, subset: str):
    """should return dict with list of tuples for each scene"""
    # assert filename.endswith('.vkd')
    # assert exists(filename)
    test_dt_dict = OrderedDict([('Scene01', [(0, 9), (15, 20)]),
                                ('Scene02', [(9, 13)]),
                                ])
    test_dt_variations = ['clone']

    whole_val_for_gen_flow_dt_dict = OrderedDict([('Scene01', [(0, 446)]),
                                                  ('Scene02', [(0, 232)]),
                                                  ('Scene06', [(0, 269)]),
                                                  ('Scene18', [(0, 338)]),
                                                  ('Scene20', [(0, 836)]),
                                                  ])
    whole_val_for_gen_flow_dt_variations = ['clone', 'overcast', 'fog', '15-deg-left', 'sunset', '15-deg-right']
    if filename == 'test_vkd':
        return test_dt_variations, test_dt_dict
    if filename == 'whole_val_for_gen_flow_vkd':
        assert subset == 'val'
        return whole_val_for_gen_flow_dt_variations, whole_val_for_gen_flow_dt_dict

    with open(filename, 'r') as f:
        variations, sequences = json.load(f)
        return variations, OrderedDict([(scene, split[subset]) for scene, split in sequences.items()])


def get_interval_len(interval):
    return interval[1] - interval[0] + 1


VKSequence = namedtuple('VKSequence', ['variation', 'scene', 'interval'])


class VkittiVideo(Dataset):
    """
    Handles the loading and batching for Virtual KITTI video sequences.

    If is_sequence_start, the number of flows is k_frame - 1.
    Otherwise, is k_frame (number of images = number of flows)
    Parameters:
        root: root dir of v_kitti
        subset: 'train', 'val', 'test'
        datalist: .vkd file with train, val, test splits
        k_frame: return group of k frames, label only for frame k
                 ex: k_frame=4, return frames(0,1,2,3) and label for frame 3
        all_labels: if True, return labels for all frames (not only for kth frame)
        include_incomplete: if True, return incomplete batches also; otherwise, return only complete batches
        pair_transform: transformer for both inputs and labels
        flow_dir: directory with cached flows
        return_last_filename: if True, return the filename for the last frame in batch
    """

    def __init__(self, root: str, subset: str, datalist: str, k_frame: int, all_labels: bool, include_incomplete: bool, pair_transform, flow_dir: str = None, return_last_filename=False):
        self.root = root
        self.original_sequences = self._make_seq_list(*read_v_kitti_sequences(datalist, subset))
        self.used_sequences = self.original_sequences
        self.k_frame = k_frame
        self.all_labels = all_labels
        self.include_incomplete = include_incomplete
        self.num_batches = self._compute_num_batches()
        self.pair_transform = pair_transform
        self.flow_dir = flow_dir
        self.return_last_filename = return_last_filename

    def shuffle(self):
        self.used_sequences = random.sample(self.original_sequences, len(self.original_sequences))

    def __getitem__(self, index):
        """index is batch index"""
        for seq in self.used_sequences:
            num_batches = self._count_contained_batches(seq.interval)
            if num_batches > index:
                # we found the needed sequence
                break
            index -= num_batches

        is_seq_start = (index == 0)
        frame_no_low = seq.interval[0] + index * self.k_frame
        basedir = join(self.root, f'{seq.scene}/{seq.variation}/frames')

        input_frames = []
        flow_frames = []
        label_frames = []

        frame_no_hi = min(frame_no_low + self.k_frame, seq.interval[1] + 1)
        for curr_frame_no in range(frame_no_low, frame_no_hi):
            curr_frame_input = join(basedir, f'rgb/Camera_0/rgb_{curr_frame_no:05}.jpg')
            input_frames.append(Image.open(curr_frame_input).convert('RGB'))

            if self.all_labels:
                curr_frame_label = join(basedir, f'classSegmentation/Camera_0/classgt_trainId_{curr_frame_no:05}.png')
                label_frames.append(Image.open(curr_frame_label).convert('L'))

            if self.flow_dir:
                if is_seq_start and curr_frame_no == frame_no_low:
                    # first frame of a start batch, there is no flow
                    continue
                ext = 'png' if self.flow_dir == 'backwardFlow' else 'tpf'
                f_name = 'backwardFlow' if self.flow_dir == 'backwardFlow' else 'backFlow'
                curr_frame_flow = join(self.root, self.flow_dir, f'{seq.scene}/{seq.variation}/frames', f'{self.flow_dir}/Camera_0/{f_name}_{curr_frame_no:05}.{ext}')
                if not exists(curr_frame_flow):
                    # curr_frame_flow = join(basedir, f'{self.flow_dir}/Camera_0/backFlow_{curr_frame_no:05}.tpf')
                    pass

                if ext == 'tpf':
                    flow = torch.load(curr_frame_flow)
                else:
                    flow = transform_v_kitti_png_flow(read_vkitti_png_flow(curr_frame_flow))
                flow_frames.append(flow)
        else:
            if not self.all_labels:
                # add label for last element, was not previously added
                curr_frame_label = join(basedir, f'classSegmentation/Camera_0/classgt_trainId_{curr_frame_no:05}.png')
                label_frames.append(Image.open(curr_frame_label).convert('L'))

        if self.pair_transform:
            input_frames, label_frames = self.pair_transform(input_frames, label_frames)

        if self.return_last_filename:
            return input_frames, label_frames, flow_frames, is_seq_start, curr_frame_label
        else:
            return input_frames, label_frames, flow_frames, is_seq_start

    def _compute_num_batches(self):
        """a batch is a sequence of k_frame frames"""
        num_batches = 0
        for seq in self.original_sequences:
            num_batches += self._count_contained_batches(seq.interval)
        return num_batches

    def _make_seq_list(self, variations, sequences: OrderedDict):
        all_seq = []
        for variation in variations:
            for scene, seq_list in sequences.items():
                for seq in seq_list:
                    all_seq.append(VKSequence(variation, scene, seq))

        return all_seq

    def _count_contained_batches(self, interval):
        length = get_interval_len(interval)
        nr = length // self.k_frame
        if self.include_incomplete and length % self.k_frame != 0:
            nr += 1

        return nr

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    loader = VkittiVideo(Config.get_datadir('v_kitti'), subset='train', datalist='resources/dataset_lists/full.vkd', k_frame=4, all_labels=False, include_incomplete=False, pair_transform=None, flow_dir=None)
    print(len(loader))
    data = DataLoader(loader, num_workers=10)

    for i in range(len(loader)):
        x = loader[i]
        print(x[3])
