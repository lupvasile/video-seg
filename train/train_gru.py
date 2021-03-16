import argparse
import time
from os import makedirs
from os.path import exists, basename, join
from shutil import copyfile

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config.config import Config
from config.config import TRAIN_RESOLUTION, VCN_MAXDISP_FAC
from dataset.dataset_metadata import get_dataset_metadata
from dataset.dataset_transforms import get_transform_video
from dataset.dataset_video import VkittiVideo, CityscapesVideo, CityscapesVideoSmall
from model.erfnet import ERFNet, non_bottleneck_1d
from model.gru import GRU, run_gru_sequence
from model.train_context import TrainContext
from model.vcn_wrapped import VCN_Wrapped
from utils.class_weights import get_class_weights
from utils.iou_meter import IoUMeter
from utils.logger import AutomatedLogger
from utils.misc import copy_object_sourcefile

device = torch.device('cuda')


def get_dataset_train(args, transform):
    datadir = Config.get_datadir(args.dataset)
    if args.dataset == 'v_kitti_video':
        return VkittiVideo(root=datadir, subset='train', datalist=args.datalist, k_frame=args.kth_frame, all_labels=False, include_incomplete=False, pair_transform=transform, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video':
        return CityscapesVideo(root=datadir, pair_transform=transform, subset='train', nr_frames=args.kth_frame, return_filename=False, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video_small':
        return CityscapesVideoSmall(root=datadir, pair_transform=transform, subset='train', nr_frames=args.kth_frame, return_filename=False, flow_dir=args.flow_dir)
    raise NotImplementedError


def get_dataset_val(args, transform):
    datadir = Config.get_datadir(args.dataset)
    if args.dataset == 'v_kitti_video':
        return VkittiVideo(root=datadir, subset='val', datalist=args.datalist, k_frame=args.batch_size, all_labels=True, include_incomplete=True, pair_transform=transform, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video':
        return CityscapesVideo(root=datadir, pair_transform=transform, subset='val', nr_frames=args.kth_frame, return_filename=False, flow_dir=args.flow_dir)
    if args.dataset == 'cityscapes_video_small':
        return CityscapesVideoSmall(root=datadir, pair_transform=transform, subset='val', nr_frames=args.kth_frame, return_filename=False, flow_dir=args.flow_dir)
    raise NotImplementedError


def get_data_loaders(args):
    transform = get_transform_video(args.dataset, height=TRAIN_RESOLUTION[args.dataset].height, width=TRAIN_RESOLUTION[args.dataset].width)

    dataset_train = get_dataset_train(args, transform)
    dataset_val = get_dataset_val(args, transform)

    pin_mem = not args.no_pin_memory
    print(f'Pin memory: {pin_mem}')

    loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=None, shuffle=False, pin_memory=pin_mem)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=None, shuffle=False, pin_memory=pin_mem)

    return loader_train, loader_val


def save_train_state(args, ctx: TrainContext, is_best):
    ctx.save_checkpoint(join(args.savedir, 'checkpoint.pth.tar'))
    ctx.save_net_state_dicts(join(args.savedir, 'model_weights'))

    if args.epochs_save > 0 and ctx.last_completed_epoch > 0 and ctx.last_completed_epoch % args.epochs_save == 0:
        ctx.save_net_state_dicts(f'{join(args.savedir, "model")}_ep_{ctx.last_completed_epoch:03}')
    if (is_best):
        ctx.save_checkpoint(join(args.savedir, 'checkpoint_best.pth.tar'))
        ctx.save_net_state_dicts(join(args.savedir, 'model_weights_best'))


def trainer(args, seg_network, flow_network, gru_cell):
    datasetMeta = get_dataset_metadata(args.dataset)
    weight = get_class_weights(args.dataset, args.datalist, False, TRAIN_RESOLUTION[args.dataset].height, TRAIN_RESOLUTION[args.dataset].width, args.num_workers, args.batch_size, args.kth_frame).to(device)
    print(f'Weights are: {weight}')
    criterion = nn.CrossEntropyLoss(ignore_index=datasetMeta.IGNORE_INDEX, reduction='mean', weight=weight)
    gru_optimizer = Adam(params=gru_cell.parameters(), lr=1e-5, betas=(0.95, 0.99), eps=1e-8)

    seg_network.requires_grad_(False)
    seg_params = []
    for module in seg_network.named_modules():
        if isinstance(module[1], non_bottleneck_1d):
            seg_params += list(module[1].parameters())
            module[1].requires_grad_(True)
    seg_optimizer = Adam(params=seg_params, lr=1e-10)

    ctx = TrainContext(seg_network=seg_network, flow_network=flow_network, gru_network=gru_cell, seg_optimizer=seg_optimizer, gru_optimizer=gru_optimizer)
    return train(args, ctx, criterion)


def train(args, ctx: TrainContext, criterion):
    ctx.max_iou = 0

    loader_train, loader_val = get_data_loaders(args)
    print(f'Dataset size train: {len(loader_train.dataset)}, val: {len(loader_val.dataset)}')

    logger = AutomatedLogger(args.savedir)

    if args.resume:
        ctx.load_checkpoint(args.resume)

    start_epoch = ctx.last_completed_epoch + 1

    datasetMeta = get_dataset_metadata(args.dataset)

    for ctx.epoch in range(start_epoch, args.num_epochs + 1):

        curr_lr = 0
        for param_group in ctx.gru_optimizer.param_groups:
            curr_lr = float(param_group['lr'])
            print("CURRENT GRU LR: ", param_group['lr'])

        train_seg = bool(args.epoch_start_seg is not None) and ctx.epoch >= args.epoch_start_seg

        loader_train.dataset.shuffle()
        time_start_epoch = time.time()
        iouTrain = IoUMeter(datasetMeta.NUM_CLASSES, datasetMeta.IGNORE_INDEX) if args.train_iou else None
        average_epoch_loss_train = do_epoch(args, ctx, loader_train, criterion, train_gru=True, train_seg=train_seg, train_flow=False, IoU=iouTrain)

        iouVal = IoUMeter(datasetMeta.NUM_CLASSES, datasetMeta.IGNORE_INDEX) if args.val_iou else None
        average_epoch_loss_val = do_epoch(args, ctx, loader_val, criterion, train_gru=False, train_seg=False, train_flow=False, IoU=iouVal)
        time_epoch = time.time() - time_start_epoch
        print(f'TRAIN + VAL duration: {time_epoch}')
        logger.write(ctx.epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain.getIoU()[0] if iouTrain else 0, iouVal.getIoU()[0] if iouVal else 0, curr_lr, time_epoch)

        if iouVal is None:
            curr_iou = -average_epoch_loss_val
        else:
            curr_iou = iouVal.getIoU()[0]

        is_best = curr_iou > ctx.max_iou
        ctx.max_iou = max(curr_iou, ctx.max_iou)
        ctx.last_completed_epoch = ctx.epoch

        save_train_state(args, ctx, is_best)

        if is_best:
            with open(args.savedir + "/best.txt", "w") as f:
                f.write(f'max val iou={iouVal.getIoU()[0] * 100 if iouVal else 0:.2f}, in epoch {ctx.epoch}')


def do_epoch(args, ctx: TrainContext, loader: DataLoader, criterion, train_gru: bool, train_seg: bool, train_flow: bool, IoU: IoUMeter):
    assert train_flow == False, 'currently flow training is not supported'

    is_validation = not (train_gru or train_seg or train_flow)
    print(f'======CURRENT EPOCH --- {ctx.epoch} --- {"TRAIN" if not is_validation else "VALIDATE"}======')

    all_outputs = is_validation and 'cityscapes' not in args.dataset
    print(f'all_outputs: {all_outputs}')
    print(f'train_gru: {train_gru}, train_seg: {train_seg}')

    batch_losses = []
    batch_times = []

    if (IoU):
        IoU.reset()

    ctx.gru_network.train(train_gru)
    ctx.seg_network.train(False)
    # for module in ctx.seg_network.named_modules():
    #   if isinstance(module[1], nn.modules.dropout._DropoutNd):
    #      module[1].eval()
    ctx.flow_network.train(train_flow)

    prev_image = prev_cell_state = None
    for step, (images, labels, flow_frames, is_seq_start) in enumerate(loader):
        images: torch.Tensor
        start_time = time.time()

        assert images.ndimension() == 4, 'currently only one batch supported'

        if is_seq_start:
            # new video sequence
            prev_image = None
            prev_cell_state = None

        images = images.to(device)
        labels = labels.to(device)

        if train_gru:
            ctx.gru_network.zero_grad()
        if train_seg:
            ctx.seg_network.zero_grad()

        if flow_frames:  # get the flow from DataLoader
            flow_frames = [f.to(device).unsqueeze(0) for f in flow_frames]
        else:  # compute the flow
            torch.set_grad_enabled(train_flow)
            flow_frames = []
            if not is_seq_start:
                flow, _ = ctx.flow_network(images[0], prev_image.squeeze(0))
                flow_frames.append(flow)

            for i in range(1, images.shape[0]):
                flow, _ = ctx.flow_network(images[i], images[i - 1])
                flow_frames.append(flow)

        torch.set_grad_enabled(train_seg)
        seg_frames = ctx.seg_network(images)

        torch.set_grad_enabled(not is_validation)
        outputs = run_gru_sequence(ctx.gru_network, seg_frames, flow_frames, images, prev_cell_state, prev_image, all_outputs=all_outputs)

        if criterion:
            loss = criterion(outputs, labels[:, 0])
        else:
            loss = 0.0

        if not is_validation:
            loss.backward()

            if train_gru:
                ctx.gru_optimizer.step()
            if train_seg:
                ctx.seg_optimizer.step()

        prev_image = images[-1].detach().clone().unsqueeze(0)
        prev_cell_state = outputs[-1].detach().clone().unsqueeze(0)

        batch_losses.append(loss.item())
        batch_times.append(time.time() - start_time)

        if (IoU):
            IoU.addBatch(outputs.max(1)[1].unsqueeze(1).detach(), labels.detach())

        if args.show_loss > 0 and step % args.show_loss == 0:
            avg = sum(batch_losses) / len(batch_losses)
            print(f'loss: {avg:0.4}, epoch: {ctx.epoch}, step: {step}, ',
                  f'Avg time/group: {sum(batch_times) / len(batch_times):.4f}, ',
                  f'Curr time/group: {batch_times[-1]:.4f}')

    average_epoch_loss = sum(batch_losses) / len(batch_losses)

    if (IoU):
        iou_val, iou_classes = IoU.getIoU()
        print(f'ep {ctx.epoch} MEAN IoU on {"TRAIN" if not is_validation else "VAL"} set: {iou_val * 100:.2f}%')

    return average_epoch_loss


def main(args):
    validate_args(args)

    if not args.no_benchmark:
        cudnn.benchmark = True

    print(f'cudnn.enabled={cudnn.enabled}, cudnn.benchmark={cudnn.benchmark}')
    start_training = time.time()

    savedir = args.savedir

    if not exists(savedir):
        makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args).replace(', ', '\n').replace('(', '(\n'))

    copyfile(__file__, savedir + '/' + basename(__file__))

    datasetMeta = get_dataset_metadata(args.dataset)

    if args.static == 'erfnet':
        seg_network = ERFNet(datasetMeta.NUM_CLASSES)
        copy_object_sourcefile(seg_network, savedir)

        seg_network = nn.DataParallel(seg_network).cuda()  # need this for now
        status = seg_network.load_state_dict(torch.load(args.static_weights), strict=False)
        print(f'segmentation model loading: {status}')
        seg_network = seg_network.module
        seg_network.to(device)

    if args.flow == 'vcn':
        flow_network = load_vcn(args.flow_weights, VCN_MAXDISP_FAC[args.dataset].maxdisp, VCN_MAXDISP_FAC[args.dataset].fac, TRAIN_RESOLUTION[args.dataset].width, TRAIN_RESOLUTION[args.dataset].height)
        copy_object_sourcefile(flow_network, savedir)
        flow_network.to(device)
    if not args.flow:
        flow_network = nn.Module().to(device)

    gru_network = GRU(conv_size=(args.gru_size, args.gru_size), nr_channels=datasetMeta.NUM_CLASSES, show_timeout=args.show_timeout)
    copy_object_sourcefile(gru_network, savedir)
    gru_network.to(device)

    trainer(args, seg_network, flow_network, gru_network)

    training_duration = time.time() - start_training
    minutes, seconds = divmod(int(training_duration), 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Training duration: {hours:02}:{minutes:02}:{seconds:02}")


def load_vcn(weights_filename, maxdisp, fac, MAX_WIDTH, MAX_HEIGHT):
    assert MAX_WIDTH > MAX_HEIGHT, f'are you sure width {MAX_WIDTH} smaller than height {MAX_HEIGHT}?'
    assert maxdisp > 100 and fac <= 2
    pretrained_dict = torch.load(weights_filename)
    mean_L = pretrained_dict['mean_L']
    mean_R = pretrained_dict['mean_R']
    flow_network = VCN_Wrapped([1, MAX_WIDTH, MAX_HEIGHT], md=[int(4 * (maxdisp / 256)), 4, 4, 4, 4], fac=fac, meanL=mean_L, meanR=mean_R)

    flow_network = nn.DataParallel(flow_network, device_ids=[0])
    flow_network.cuda()
    test_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if not ('grid' not in k and (('flow_reg' not in k) or ('conv1' in k)))}
    if test_dict:
        raise RuntimeError(f'error when loading flow module, test_dict should be empty, has length {len(test_dict)}.'
                           f'See submission.py')
    res = flow_network.load_state_dict(pretrained_dict['state_dict'], strict=False)
    if len(res.missing_keys) > 17:
        raise RuntimeError(f'failed at loading flow module, has {len(res.missing_keys)} missing keys')
    flow_network = flow_network.module
    return flow_network


def validate_args(args):
    assert not args.savedir.endswith('/'), 'savedir should not end in /'
    assert args.kth_frame >= 1 and args.kth_frame <= 20, "The number of frames must be between 1 and 20."
    assert bool(args.flow) or bool(args.flow_dir), "Should specify flow or flow-dir options"
    assert bool(args.flow) != bool(args.flow_dir), "Can not specify both flow and flow-dir options"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train GRU')

    parser.add_argument('--static', choices=['erfnet'], help='Which static network to use.', default='erfnet')
    parser.add_argument('--flow', choices=['vcn'], help='Which optical flow method to use.')
    parser.add_argument('--flow-dir', default='flow_vcn_kitti')
    parser.add_argument('--kth-frame', type=int, help='In training, consider labels for every kth frame', default=4, required=False)
    parser.add_argument('--dataset', choices=['cityscapes_video', 'cityscapes_video_small', 'v_kitti_video'], default='v_kitti_video')
    parser.add_argument('--datalist', default='resources/dataset_lists/full.vkd', help='v_kitti dataset split .vkd file')

    parser.add_argument('--static-weights', default='resources/save/v_kitti_erfnet_2/model_best.pth')
    parser.add_argument('--flow-weights', default='resources/flow_weights/kitti-ft-trainval/finetune_149999.tar')
    parser.add_argument('--gru-size', default=7)

    parser.add_argument('--resume', help='path to a .tar file with training context')

    parser.add_argument('--show-loss', type=int, default=200)
    parser.add_argument('--savedir', default='resources/save_grfp/tests')
    parser.add_argument('--epochs-save', type=int, default=0, help='save every n epochs')

    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--epoch-start-seg', type=int, default=None, help='from what epoch to train segmentation network, if None segmentation is not trained')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=12, help='validation sequence batch size')

    parser.add_argument('--train-iou', action='store_true', default=False)
    parser.add_argument('--val-iou', action='store_true', default=True)

    parser.add_argument('--show-timeout', type=int, default=None, help='display the processed images and refinement steps. When not specified,'
                                                                       'no image is shown, when is 0, press any key to see the next image, '
                                                                       'when an integer, each image is displayed for this amount of milliseconds.')

    args = parser.parse_args()

    main(args)
