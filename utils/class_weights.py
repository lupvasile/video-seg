import torch
from torch.utils.data.dataloader import DataLoader

from config.config import TRAIN_RESOLUTION
from dataset.dataset_metadata import get_dataset_metadata
from dataset.dataset_seg import get_dataset_labels_only
from dataset.dataset_transforms import get_transform_labels_only


def get_class_weights(dataset_name, data_list, enc, height, width, num_workers, batch_size, video_train_kth_frame=None):
    dataset_meta = get_dataset_metadata(dataset_name)
    transform = get_transform_labels_only(dataset_name, enc, height, width)
    dataset = get_dataset_labels_only(dataset_name, data_list, transform, 'train', video_train_kth_frame)

    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    hist = torch.zeros(dataset_meta.NUM_CLASSES, dtype=torch.float32)

    for labels in loader:
        hist = hist + torch.histc(labels.float(), dataset_meta.NUM_CLASSES, 0, dataset_meta.NUM_CLASSES - 1)

    hist = hist / hist.sum()

    weights = torch.log1p(hist + 0.1).reciprocal()

    if dataset_meta.IGNORE_INDEX:
        weights[dataset_meta.IGNORE_INDEX] = 0.0

    return weights


if __name__ == '__main__':
    dataset = 'cityscapes_video'
    datalist = 'resources/dataset_lists/full.vkd'
    x = get_class_weights(dataset, datalist, False, TRAIN_RESOLUTION[dataset].height, TRAIN_RESOLUTION[dataset].width, 4, 10)
    print(x)
