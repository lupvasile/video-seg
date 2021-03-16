import argparse
import time
from os import makedirs
from os.path import basename, join, exists
from shutil import copyfile

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader

import train.train_gru as train_gru
from config.config import Config, TRAIN_RESOLUTION, VCN_MAXDISP_FAC
from dataset.dataset_transforms import get_transform_video
from dataset.dataset_video import CityscapesVideo, CityscapesVideoSmall
from utils.misc import copy_object_sourcefile

device = torch.device('cuda')


def get_dataset(args, transform, subset):
    datadir = Config.get_datadir(args.dataset)
    if args.dataset == 'cityscapes_video':
        return CityscapesVideo(root=datadir, pair_transform=transform, subset=subset, nr_frames=args.frames, return_filename=True, flow_dir=None)
    if args.dataset == 'cityscapes_video_small':
        return CityscapesVideoSmall(root=datadir, pair_transform=transform, subset=subset, nr_frames=args.frames, return_filename=True, flow_dir=None)


def get_DataLoader(args, subset):
    transform = get_transform_video(args.dataset, height=TRAIN_RESOLUTION[args.dataset].height, width=TRAIN_RESOLUTION[args.dataset].width)
    dataset = get_dataset(args, transform, subset)

    pin_mem = not args.no_pin_memory
    print(f'Pin memory: {pin_mem}')
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None, shuffle=False, pin_memory=pin_mem)

    return loader


def gen_flow(args, flow_network: nn.Module, subset: str):
    data_loader = get_DataLoader(args, subset)
    print(f'Dataset size train: {len(data_loader.dataset)}')
    print(f'Subset: {subset}')

    flow_network.eval()
    torch.set_grad_enabled(False)

    for step, (images, _, flow_frames, is_seq_start, label_filename) in enumerate(data_loader):
        images: torch.Tensor
        start_time = time.time()

        assert images.ndimension() == 4, 'currently only one batch supported'
        assert (not flow_frames), 'flow already existent at specified location'
        assert is_seq_start == True, 'for cityscapes is_seq_start should always be true'

        images = images.to(device)

        for i in range(1, args.frames):
            parts = label_filename.replace('\\', '/').split('/')[-1].split('_')
            city, seq, frame = parts[0], parts[1], parts[2]
            basedir = join(args.savedir, subset, city)
            flow_filename = join(basedir, f'{city}_{seq}_{int(frame) - (args.frames - i) + 1:06}_backFlow.tpf')

            if exists(flow_filename):
                if args.skip_existing:
                    print(f'WARNING: {basename(flow_filename)} already existing. skipping...')
                    continue
                else:
                    print(f'WARNING: {basename(flow_filename)} already existing. overwriting...')

            if not exists(basedir):
                makedirs(basedir)

            flow, _ = flow_network(images[i], images[i - 1])
            flow = flow.cpu().detach().half().squeeze(0)

            torch.save(flow, flow_filename)

        duration = time.time() - start_time

        print(f'step: {step}, time: {duration:.3f}')


def main(args):
    validate_args(args)

    if not args.no_benchmark:
        cudnn.benchmark = True

    print(f'cudnn.enabled={cudnn.enabled}, cudnn.benchmark={cudnn.benchmark}')
    start_inference = time.time()

    args.savedir = join(Config.get_datadir(args.dataset), args.savedir)
    makedirs(args.savedir, exist_ok=True)
    savedir = args.savedir
    print(f'Saving to: {savedir}')

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args).replace(', ', '\n').replace('(', '(\n'))

    copyfile(__file__, join(savedir, basename(__file__)))

    if args.flow == 'vcn':
        flow_network = train_gru.load_vcn(args.flow_weights, VCN_MAXDISP_FAC[args.dataset].maxdisp, VCN_MAXDISP_FAC[args.dataset].fac, TRAIN_RESOLUTION[args.dataset].width, TRAIN_RESOLUTION[args.dataset].height)
        copy_object_sourcefile(flow_network, savedir)
        flow_network.to(device)

    gen_flow(args, flow_network, 'train')
    gen_flow(args, flow_network, 'val')

    inference_duration = time.time() - start_inference
    minutes, seconds = divmod(int(inference_duration), 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Flow duration: {hours:02}:{minutes:02}:{seconds:02}")


def validate_args(args):
    assert not args.savedir.endswith('/'), 'savedir should not end in /'
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate flow files')

    parser.add_argument('--flow', choices=['vcn'], help='Which optical flow method to use.', default='vcn')
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)
    parser.add_argument('--dataset', choices=['cityscapes_video', 'cityscapes_video_small'], default='cityscapes_video')
    parser.add_argument('--flow-weights', default='resources/flow_weights/kitti-ft-trainval/finetune_149999.tar')

    parser.add_argument('--savedir', default='flow_vcn_kitti', help='savedir will be appended to dataset root path')

    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--skip-existing', action='store_true')

    args = parser.parse_args()

    main(args)
