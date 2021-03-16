from argparse import ArgumentParser
from os.path import exists

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import Config, TRAIN_RESOLUTION
from dataset.dataset_metadata import get_dataset_metadata
from dataset.dataset_seg import get_dataset
from dataset.dataset_transforms import GenericTransformSegEval
from model.erfnet import ERFNet
from utils.iou_meter import IoUMeter

device = torch.device('cuda')


def main(args):
    dataset_meta = get_dataset_metadata(args.dataset)

    weights_path = args.weights
    assert exists(weights_path), f'{weights_path} does not exist'

    model = ERFNet(dataset_meta.NUM_CLASSES)
    model = torch.nn.DataParallel(model).to(device)

    status = model.load_state_dict(torch.load(weights_path), strict=False)
    print(status)

    model.eval()
    torch.set_grad_enabled(False)

    res = TRAIN_RESOLUTION[args.dataset + '_video']
    transform = GenericTransformSegEval(resolution_input=res, resolution_target=None if args.orig_res else res, relabel_target_val=dataset_meta.IGNORE_INDEX)
    dataset = get_dataset(args.dataset, args.datalist, transform, args.subset)
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    loader = tqdm(loader)

    iou = IoUMeter(dataset_meta.NUM_CLASSES, dataset_meta.IGNORE_INDEX, resize_first_batch=args.orig_res)
    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        outputs = outputs.max(1)[1].unsqueeze(1)
        iou.addBatch(outputs, labels)

    iou_mean, iou_class = iou.getIoU()

    print("IoU per class:")
    for iou_class_val, class_name in zip(iou_class, dataset_meta.CLASS_NAMES):
        print(f'{iou_class_val * 100:.2f}% {class_name}')
    print("=======================================")
    print(f'IoU mean: {iou_mean * 100:.2f}%')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--weights', default="resources/save/v_kitti_erfnet_2/model_best.pth")

    parser.add_argument('--dataset', choices=Config.SUPPORTED_DATASETS, default='v_kitti')
    parser.add_argument('--datalist', default='resources/dataset_lists/full.vkd', help='v_kitti dataset split .vkd file')
    parser.add_argument('--subset', default='val')

    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--orig-res', action='store_true', help='if specified, report result on original target resolution', default=False)

    parsed_args = parser.parse_args()
    main(parsed_args)
