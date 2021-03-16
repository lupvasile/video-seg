import argparse
import time
from os.path import join

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import train.train_gru as train_gru
from config.config import Config
from config.config import TRAIN_RESOLUTION, VCN_MAXDISP_FAC
from dataset.dataset_metadata import get_dataset_metadata
from dataset.dataset_transforms import GenericVideoTransform
from dataset.dataset_video import VkittiVideo, CityscapesVideo, CityscapesVideoSmall
from model.erfnet import ERFNet
from model.gru import GRU
from model.train_context import TrainContext
from model.vcn_wrapped import VCN_Wrapped
from utils.iou_meter import IoUMeter

device = torch.device('cuda')


def get_dataset(args, transform, subset):
    datadir = Config.get_datadir(args.dataset)
    if args.dataset == 'v_kitti_video':
        return VkittiVideo(root=datadir, subset=subset, datalist=args.datalist, k_frame=args.batch_size, all_labels=True, include_incomplete=True, pair_transform=transform, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video':
        return CityscapesVideo(root=datadir, pair_transform=transform, subset=subset, nr_frames=args.frames, return_filename=False, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video_small':
        return CityscapesVideoSmall(root=datadir, pair_transform=transform, subset=subset, nr_frames=args.frames, return_filename=False, flow_dir=args.flow_dir)
    raise NotImplementedError


def get_DataLoader(args):
    res = TRAIN_RESOLUTION[args.dataset]
    datasetMeta = get_dataset_metadata(args.dataset)
    transform = GenericVideoTransform(resolution_input=res, resolution_target=None if args.orig_res else res, relabel_target_val=datasetMeta.IGNORE_INDEX)
    dataset = get_dataset(args, transform, args.subset)

    pin_mem = not args.no_pin_memory
    print(f'Pin memory: {pin_mem}')
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None, shuffle=False, pin_memory=pin_mem)

    return loader


def evaluate(args, seg_network, flow_network, gru_cell):
    ctx = TrainContext(seg_network=seg_network, flow_network=flow_network, gru_network=gru_cell, seg_optimizer=None, gru_optimizer=None)
    ctx.load_net_state_dicts(args.weights_dir, args.weights_name)

    dataset_meta = get_dataset_metadata(args.dataset)
    iou = IoUMeter(dataset_meta.NUM_CLASSES, dataset_meta.IGNORE_INDEX, resize_first_batch=args.orig_res)

    loader = get_DataLoader(args)
    print(f'Dataset size train: {len(loader.dataset)}')
    print(f'Subset: {args.subset}')
    loader = tqdm(loader)

    dummy_criterion = lambda _, t: torch.tensor(0.0, dtype=torch.float32, device=t.device)

    train_gru.do_epoch(args, ctx, loader, criterion=dummy_criterion, train_gru=False, train_seg=False, train_flow=False, IoU=iou)

    iou_mean, iou_class = iou.getIoU()

    print("IoU per class:")
    for iou_class_val, class_name in zip(iou_class, dataset_meta.CLASS_NAMES):
        print(f'{iou_class_val * 100:.2f}% {class_name}')
    print("=======================================")
    print(f'IoU mean: {iou_mean * 100:.2f}%')

    if not args.no_out_file:
        out_file = join(args.weights_dir, f'acc_{args.subset}_{"orig_" if args.orig_res else ""}{iou_mean * 100:.2f}.txt')
        with open(out_file, 'w') as f:
            for i in range(iou_class.shape[0]):
                f.write(f'{iou_class[i] * 100:.2f} {dataset_meta.CLASS_NAMES[i]}\n')
            f.write(f'{iou_mean * 100:.2f} MEAN')


def main(args):
    validate_args(args)

    if not args.no_benchmark:
        cudnn.benchmark = True

    print(f'cudnn.enabled={cudnn.enabled}, cudnn.benchmark={cudnn.benchmark}')
    start_eval = time.time()

    datasetMeta = get_dataset_metadata(args.dataset)

    if args.static == 'erfnet':
        seg_network = ERFNet(datasetMeta.NUM_CLASSES)
        seg_network.to(device)

    if args.flow == 'vcn':
        res = TRAIN_RESOLUTION[args.dataset]
        vcn_param = VCN_MAXDISP_FAC[args.dataset]
        flow_network = VCN_Wrapped([1, res.width, res.height], md=[int(4 * (vcn_param.maxdisp / 256)), 4, 4, 4, 4], fac=vcn_param.fac, meanL=None, meanR=None)
        flow_network.to(device)
    if not args.flow:
        flow_network = nn.Module().to(device)

    gru_network = GRU(conv_size=(args.gru_size, args.gru_size), nr_channels=datasetMeta.NUM_CLASSES, show_timeout=args.show_timeout)
    gru_network.to(device)

    evaluate(args, seg_network, flow_network, gru_network)

    training_duration = time.time() - start_eval
    minutes, seconds = divmod(int(training_duration), 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Evaluation duration: {hours:02}:{minutes:02}:{seconds:02}")


def validate_args(args):
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    assert bool(args.flow) or bool(args.flow_dir), "Should specify flow or flow-dir options"
    assert bool(args.flow) != bool(args.flow_dir), "Can not specify both flow and flow-dir options"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate grfp')

    parser.add_argument('--static', choices=['erfnet'], help='Which static network to use.', default='erfnet')
    parser.add_argument('--flow', choices=['vcn'], help='Which optical flow method to use.')
    parser.add_argument('--flow-dir', default='backwardFlow')

    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=4, required=False)

    parser.add_argument('--dataset', choices=['cityscapes_video', 'cityscapes_video_small', 'v_kitti_video'], default='v_kitti_video')
    parser.add_argument('--datalist', default='resources/dataset_lists/full.vkd', help='v_kitti dataset split .vkd file')
    parser.add_argument('--subset', default='val', choices=['train','val','test'])

    parser.add_argument('--weights-dir', default='resources/save_grfp/train_vk_7x7_backFlow', help='directory containing model weights')
    parser.add_argument('--gru-size', default=7)
    parser.add_argument('--weights-name', default='model_weights_best', help='filename of .pth files')

    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=12, help='validation sequence batch size')

    parser.add_argument('--orig-res', action='store_true', help='if specified, report result on original target resolution', default=False)
    parser.add_argument('--no-out-file', action='store_true')

    parser.add_argument('--show-timeout', type=int, default=None, help='display the processed images and refinement steps. When not specified,'
                                                                       'no image is shown, when is 0, press any key to see the next image, '
                                                                       'when an integer, each image is displayed for this amount of milliseconds.')

    args = parser.parse_args()
    args.show_loss = 0

    main(args)
