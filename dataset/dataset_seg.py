import os
from os.path import exists, join, splitext, expanduser

from PIL import Image
from torch.utils.data import Dataset

from config.config import Config
from dataset.dataset_video import read_v_kitti_sequences


class _Cityscapes(Dataset):

    def __init__(self, data_root, subset='train', pair_transform=None, input_transform=None, target_transform=None):
        """order of operations: input_tranform on input, target_transform on labels, pair_transform on both"""

        self.images_root = join(data_root, 'leftImg8bit/')
        self.labels_root = join(data_root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        assert exists(self.images_root), f'{self.images_root} does not exist'
        assert exists(self.labels_root), f'{self.labels_root} does not exist'

        self.filenames_labels = [join(root, f) for root, _, files in os.walk(expanduser(self.labels_root)) for f in
                                 files if f.endswith("_labelTrainIds.png")]
        self.filenames_labels.sort()

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename_label = self.filenames_labels[index]
        label_image = Image.open(filename_label).convert('L')

        parts = filename_label.replace('\\', '/').split('/')[-1].split('_')
        city, seq, frame = parts[0], parts[1], parts[2]

        filename_input = join(self.images_root, city, f'{city}_{seq}_{frame}_leftImg8bit.png')
        input_image = Image.open(filename_input).convert('RGB')

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            label_image = self.target_transform(label_image)
        if self.pair_transform:
            input_image, label_image = self.pair_transform(input_image, label_image)

        return input_image, label_image

    def __len__(self):
        return len(self.filenames_labels)


class _CityscapesLabelOnly(_Cityscapes):

    def __init__(self, data_root, subset='train', target_transform=None):
        super().__init__(data_root, subset=subset, pair_transform=None, input_transform=None, target_transform=target_transform)

    def __getitem__(self, index):
        filename_label = self.filenames_labels[index]

        label_image = Image.open(filename_label).convert('L')

        if self.target_transform:
            label_image = self.target_transform(label_image)

        return label_image


def read_dataset_list(list_filename):
    assert exists(list_filename)

    with open(list_filename, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines() if not line.strip().startswith('#')]
        return lines


class _Vkitti(Dataset):

    def __init__(self, data_root, data_list, subset='train', pair_transform=None, input_transform=None, target_transform=None, video_train_kth_frame=None):
        self.data_root = data_root

        if data_list.endswith('vkd'):
            self.filenames = self.extract_vkd_filenames(data_root, data_list, subset, video_train_kth_frame)
        else:
            name, ext = splitext(data_list)
            data_list = f'{name}_{subset}{ext}'
            self.filenames = [(join(data_root, data), join(data_root, label)) for data, label in
                              read_dataset_list(data_list)]

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    @staticmethod
    def extract_vkd_filenames(data_root, data_list, subset, train_kth_frame):
        """
            train_kth_frame - for train dataset, only consider the kth frame.
            ex: if k_th frame is 4, from sequence 0..10 return frames 3, 7
        """
        if subset == 'train':
            assert train_kth_frame is not None
            kth_frame = train_kth_frame
        else:
            kth_frame = 1

        variations, sequences = read_v_kitti_sequences(data_list, subset)
        filenames = []
        for variation in variations:
            for scene, seq_list in sequences.items():
                basedir = join(data_root, f'{scene}/{variation}/frames')
                for seq in seq_list:
                    for idx in range(seq[0] + kth_frame - 1, seq[1] + 1, kth_frame):
                        filenames.append((join(basedir, f'rgb/Camera_0/rgb_{idx:05}.jpg'),
                                          join(basedir, f'classSegmentation/Camera_0/classgt_trainId_{idx:05}.png')))

        return filenames

    def __getitem__(self, index):
        filename_data, filename_label = self.filenames[index]

        input_image = Image.open(filename_data).convert('RGB')
        label_image = Image.open(filename_label).convert('L')

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            label_image = self.target_transform(label_image)
        if self.pair_transform:
            input_image, label_image = self.pair_transform(input_image, label_image)

        return input_image, label_image

    def __len__(self):
        return len(self.filenames)


class _VkittiLabelOnly(_Vkitti):

    def __init__(self, data_root, data_list, subset='train', target_transform=None, video_train_kth_frame=None):
        super().__init__(data_root, data_list, subset=subset, pair_transform=None, input_transform=None, target_transform=target_transform, video_train_kth_frame=video_train_kth_frame)

    def __getitem__(self, index):
        _, filename_label = self.filenames[index]
        label = Image.open(filename_label).convert('L')

        if self.target_transform:
            label = self.target_transform(label)

        return label


def get_dataset(dataset_name, data_list, pair_transform, subset, video_train_kth_frame=None):
    assert subset in ['train', 'val', 'test']

    datadir = Config.get_datadir(dataset_name)
    if dataset_name == 'cityscapes':
        return _Cityscapes(datadir, subset, pair_transform)
    if dataset_name == 'v_kitti':
        return _Vkitti(datadir, data_list, subset, pair_transform, video_train_kth_frame=video_train_kth_frame)

    raise NotImplementedError


def get_dataset_labels_only(dataset_name, data_list, target_transform, subset, video_train_kth_frame=None):
    assert subset in ['train', 'val', 'test']

    datadir = Config.get_datadir(dataset_name)
    if 'cityscapes' in dataset_name:
        return _CityscapesLabelOnly(datadir, subset, target_transform)
    if 'v_kitti' in dataset_name:
        return _VkittiLabelOnly(datadir, data_list, subset, target_transform, video_train_kth_frame)

    raise NotImplementedError
