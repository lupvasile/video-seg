from collections import namedtuple

from .paths import *

config_shared = {'dir_v_kitti': '/media/vasi/Elements/datasets/v_kitti/'}

config_debug = {'one_batch': True, **config_shared}
config_normal = {'one_batch': False, **config_shared}


class Config:
    SUPPORTED_DATASETS = ['cityscapes', 'cityscapes_small', 'v_kitti']

    def __init__(self, config_dict):
        for k, v in config_dict.items():
            self.__setattr__(k, v)

    @classmethod
    def get_config(cls):
        return Config(config_debug)

    @classmethod
    def get_datadir(cls, dataset):
        if 'cityscapes' in dataset:
            return DIR_CITYSCAPES
        if 'v_kitti' in dataset:
            return DIR_V_KITTI
        raise ValueError(f"Dataset not known, {dataset}")


Resolution = namedtuple('Resolution', ['height', 'width'])  # for vcn, width and height must be divisible by 64
MaxDispFac = namedtuple('MaxDispFac', ['maxdisp', 'fac'])
TRAIN_RESOLUTION = {'cityscapes_video':       Resolution(height=512, width=1024),
                    'cityscapes_video_small': Resolution(height=512, width=1024),
                    'v_kitti_video':          Resolution(height=192, width=640), }
VCN_MAXDISP_FAC = {'cityscapes_video':       MaxDispFac(maxdisp=512, fac=2),
                   'cityscapes_video_small': MaxDispFac(maxdisp=512, fac=2),
                   'v_kitti_video':          MaxDispFac(maxdisp=256, fac=2), }
