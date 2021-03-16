import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import Resize, ToTensor, Compose

from config.config import TRAIN_RESOLUTION, Resolution
from dataset.dataset_metadata import get_dataset_metadata


class Relabel:
    # https://github.com/Eromera/erfnet_pytorch
    def __init__(self, old_label, new_label):
        self.old_label = old_label
        self.new_label = new_label

    def __call__(self, t):
        t[t == self.old_label] = self.new_label
        return t


class ToLabel:
    # https://github.com/Eromera/erfnet_pytorch
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).long().unsqueeze(0)


class _SegmentationTransform():
    def __init__(self, enc, augment, height, width, relabel_target_val):
        self.enc = enc
        self.augment = augment
        self.height = height
        self.width = width

        self.resize_input = Resize((self.height, self.width), Image.BILINEAR)
        self.resize_target = Resize((self.height, self.width), Image.NEAREST)
        self.resize_enc = Resize(int(self.height / 8), Image.NEAREST)
        self.to_tensor = ToTensor()
        self.finish_target = Compose([ToLabel(), Relabel(255, relabel_target_val)])

    def __call__(self, input, target):
        input = self.resize_input(input)
        target = self.resize_target(target)

        if (self.augment):
            # https://github.com/Eromera/erfnet_pytorch
            # Random horizontal flip
            p = random.random()
            if (p < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = self.to_tensor(input)
        if (self.enc):
            target = self.resize_enc(target)
        target = self.finish_target(target)

        return input, target


class GenericVideoTransform():
    def __init__(self, resolution_input: Resolution, resolution_target: Resolution, relabel_target_val):
        input_tr_list = []
        if resolution_input:
            input_tr_list.append(Resize((resolution_input.height, resolution_input.width), interpolation=Image.BILINEAR))
        input_tr_list.append(ToTensor())
        self.input_transform = Compose(input_tr_list)

        target_tr_list = []
        if resolution_target:
            target_tr_list.append(Resize((resolution_target.height, resolution_target.width), interpolation=Image.NEAREST))
        target_tr_list.append(ToLabel())
        if relabel_target_val is not None:
            target_tr_list.append(Relabel(255, relabel_target_val))
        self.target_transform = Compose(target_tr_list)

    def __call__(self, inputs, targets):
        inputs = list(map(self.input_transform, inputs))
        targets = list(map(self.target_transform, targets))

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return inputs, targets


class GenericTransformSegEval(GenericVideoTransform):
    def __init__(self, resolution_input: Resolution, resolution_target: Resolution, relabel_target_val):
        super().__init__(resolution_input, resolution_target, relabel_target_val)

    def __call__(self, input, target):
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target


def get_transform_seg(dataset_name, enc, augment, height, width):
    dataMeta = get_dataset_metadata(dataset_name)
    return _SegmentationTransform(enc=enc, augment=augment, height=height, width=width, relabel_target_val=dataMeta.IGNORE_INDEX)


def get_transform_video(dataset_name, height, width):
    dataMeta = get_dataset_metadata(dataset_name)
    res = Resolution(height, width)
    return GenericVideoTransform(resolution_input=res, resolution_target=res, relabel_target_val=dataMeta.IGNORE_INDEX)


class _LabelTransform:
    """all 255 values are relabeled to relabel_value"""

    def __init__(self, enc, height, width, relabel_value):
        self.enc = enc
        self.height = height

        if width is not None:
            self.size = (height, width)
        else:
            self.size = height

        self.relabel_value = relabel_value

    def __call__(self, target):
        target = Resize(self.size, Image.NEAREST)(target)
        if self.enc:
            target = Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, self.relabel_value)(target)

        return target


def get_transform_labels_only(dataset_name, enc, height, width):
    dataMeta = get_dataset_metadata(dataset_name)
    return _LabelTransform(enc, height, width, dataMeta.IGNORE_INDEX)


def transform_v_kitti_png_flow(read_flow):
    pred_res = Resolution(height=read_flow.shape[0], width=read_flow.shape[1])
    target_res = TRAIN_RESOLUTION['v_kitti_video']

    flow = cv2.resize(read_flow, (target_res.width, target_res.height))
    flow[:, :, 0] *= target_res.width / pred_res.width
    flow[:, :, 1] *= target_res.height / pred_res.height

    flow = ToTensor()(flow)
    return flow
