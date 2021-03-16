from collections import namedtuple

DatasetLabel = namedtuple('DatasetLabel', ['NAME', 'COLOR', 'TRAIN_ID', 'WEIGHT_ENC', 'WEIGHT_DEC'])
_DatasetMetadata = namedtuple('_DatasetMetadata', ['NUM_CLASSES', 'IGNORE_INDEX', 'CLASSES'])
# IGNORE_INDEX should be None if no index is ignored

_cityscapesClasses = [
    DatasetLabel("road",          (128, 64, 128),        0,     2.365,        2.814),
    DatasetLabel("sidewalk",      (244, 35, 232),        1,     4.423,        6.985),
    DatasetLabel("building",      (70, 70, 70),          2,     2.969,        3.789),
    DatasetLabel("wall",          (102, 102, 156),       3,     5.344,        9.942),
    DatasetLabel("fence",         (190, 153, 153),       4,     5.298,        9.770),
    DatasetLabel("pole",          (153, 153, 153),       5,     5.227,        9.511),
    DatasetLabel("traffic light", (250, 170, 30),        6,     5.439,       10.311),
    DatasetLabel("traffic sign",  (220, 220, 0),         7,     5.365,       10.026),
    DatasetLabel("vegetation",    (107, 142, 35),        8,     3.417,        4.632),
    DatasetLabel("terrain",       (152, 251, 152),       9,     5.241,        9.560),
    DatasetLabel("sky",           (70, 130, 180),       10,     4.737,        7.869),
    DatasetLabel("person",        (220, 20, 60),        11,     5.228,        9.516),
    DatasetLabel("rider",         (255, 0, 0),          12,     5.455,       10.373),
    DatasetLabel("car",           (0, 0, 142),          13,     4.301,        6.661),
    DatasetLabel("truck",         (0, 0, 70),           14,     5.426,       10.260),
    DatasetLabel("bus",           (0, 60, 100),         15,     5.433,       10.287),
    DatasetLabel("train",         (0, 80, 100),         16,     5.433,       10.289),
    DatasetLabel("motorcycle",    (0, 0, 230),          17,     5.463,       10.405),
    DatasetLabel("bicycle",       (119, 11, 32),        18,     5.394,       10.138),
    DatasetLabel("unlabeled",     (0, 0, 0),           255,     0.0,            0.0),
]
#IGNORE_INDEX is 19, not 255
_cityscapesMetadata = _DatasetMetadata(20, 19, _cityscapesClasses)

vkittyClasses = [
    DatasetLabel('terrain',       (210, 0, 200),         0,     None,                  None),
    DatasetLabel('sky',           (90, 200, 255),        1,     None,                  None),
    DatasetLabel('tree',          (0, 199, 0),           2,     None,                  None),
    DatasetLabel('vegetation',    (90, 240, 0),          3,     None,                  None),
    DatasetLabel('building',      (140, 140, 140),       4,     None,                  None),
    DatasetLabel('road',          (100, 60, 100),        5,     None,                  None),
    DatasetLabel('guardrail',     (250, 100, 255),       6,     None,                  None),
    DatasetLabel('traffic sign',  (255, 255, 0),         7,     None,                  None),
    DatasetLabel('traffic light', (200, 200, 0),         8,     None,                  None),
    DatasetLabel('pole',          (255, 130, 0),         9,     None,                  None),
    DatasetLabel('misc',          (80, 80, 80),         10,     None,                  None),
    DatasetLabel('truck',         (160, 60, 60),        11,     None,                  None),
    DatasetLabel('car',           (255, 127, 80),       12,     None,                  None),
    DatasetLabel('van',           (0, 139, 139),        13,     None,                  None),
    DatasetLabel('undefined',     (0, 0, 0),           255,     None,                  None),
]
_v_kittiMetadata = _DatasetMetadata(15, 14, vkittyClasses)

class DatasetMetadata:
    def __init__(self, datasetMeta):
        self.NUM_CLASSES = datasetMeta.NUM_CLASSES
        self.IGNORE_INDEX = datasetMeta.IGNORE_INDEX
        self.CLASS_NAMES =     [c.NAME        for c in datasetMeta.CLASSES]
        self.CLASS_COLORS =    [c.COLOR       for c in datasetMeta.CLASSES]
        self.CLASS_TRAIN_IDS = [c.TRAIN_ID    for c in datasetMeta.CLASSES]
        self.WEIGHT_ENC =      [c.WEIGHT_ENC  for c in datasetMeta.CLASSES]
        self.WEIGHT_DEC =      [c.WEIGHT_DEC  for c in datasetMeta.CLASSES]
        self.CLASSES = datasetMeta.CLASSES

_cityMeta = DatasetMetadata(_cityscapesMetadata)
_v_kittiMeta = DatasetMetadata(_v_kittiMetadata)

def get_dataset_metadata(dataset_name):
    if 'cityscapes' in dataset_name:
        return _cityMeta
    if 'v_kitti' in dataset_name:
        return _v_kittiMeta
    raise RuntimeError
