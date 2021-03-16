import cv2
import numpy as np
import torch
from utils.flowlib import flow_to_image

from dataset.dataset_metadata import get_dataset_metadata


def segmentation_to_image(seg):
    """seg should be np.array of form height x width x nr_classes"""
    assert seg.shape[2] < seg.shape[0] < seg.shape[1]

    if seg.shape[2] == 15:
        dataset_meta = get_dataset_metadata('v_kitti')
    elif seg.shape[2] == 20:
        dataset_meta = get_dataset_metadata('cityscapes')
    else:
        dataset_meta = None

    seg = seg.argmax(axis=2)
    res = np.zeros((*seg.shape, 3), dtype=np.uint8)  # rgb image

    for cls in dataset_meta.CLASSES:
        res[seg == cls.TRAIN_ID] = cls.COLOR

    return res


def segmentation_to_label(seg):
    """seg should be np.array of form height x width x nr_classes"""
    assert seg.shape[2] < seg.shape[0] < seg.shape[1]

    if seg.shape[2] == 15:
        dataset_meta = get_dataset_metadata('v_kitti')
    elif seg.shape[2] == 20:
        dataset_meta = get_dataset_metadata('cityscapes')
    else:
        dataset_meta = None

    seg = seg.argmax(axis=2)
    res = np.empty(seg.shape, dtype=np.object)  # rgb image

    for cls in dataset_meta.CLASSES:
        res[seg == cls.TRAIN_ID] = cls.NAME

    return res


class ImageViewer():
    def __init__(self):
        self.windows = set()
        self.shownImages = {}

    def onClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'pixel{y, x}\n')
            for k in sorted(self.shownImages.items()):
                print(k[0], ': ', k[1][y, x])

    def showImage(self, slotNum, img, img_type='rgb', imgtitle='undef'):
        assert img_type in ['rgb', 'flow', 'seg']
        if type(img) is torch.Tensor:
            img: torch.Tensor
            if img.ndimension() == 4:
                assert img.shape[0] == 1
                img = img.squeeze(0).permute(1, 2, 0)
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                img = img.permute(1, 2, 0)

            img = img.detach().cpu().numpy()

        # winHorizSpace = 340
        winHorizSpace = img.shape[1] + 0
        winVertSpace = img.shape[0] + 78
        # winLeftPos = (1920, 40)
        winLeftPos = (0, 0)
        nrWinsPerHoriz = 3

        winName = str(slotNum) + ' ' + imgtitle
        if img_type == 'seg':
            self.shownImages[winName] = segmentation_to_label(img)
        else:
            self.shownImages[winName] = img.copy()

        if winName not in self.windows:
            self.windows.add(winName)
            cv2.namedWindow(winName)
            cv2.moveWindow(winName, slotNum % nrWinsPerHoriz * winHorizSpace + winLeftPos[0],
                           slotNum // nrWinsPerHoriz * winVertSpace + winLeftPos[1])
            cv2.setMouseCallback(winName, self.onClick)
        if img_type == 'rgb':
            img = img[:, :, ::-1]
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
        if img_type == 'flow':
            img = flow_to_image(img)[:, :, ::-1]
        if img_type == 'seg':
            img = segmentation_to_image(img)[:, :, ::-1]

        cv2.imshow(winName, img)

    def show(self, t=None):
        if t:
            key = cv2.waitKey(t)
        else:
            key = cv2.waitKey()

        if key == 113:  # key q
            exit(21)


IM = ImageViewer()
