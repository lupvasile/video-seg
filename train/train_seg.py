import os
import time
from argparse import ArgumentParser
from os import makedirs
from os.path import basename, exists
from shutil import copyfile

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from config.config import TRAIN_RESOLUTION
from dataset.dataset_metadata import get_dataset_metadata
from dataset.dataset_seg import get_dataset
from dataset.dataset_transforms import get_transform_seg
from model.erfnet import ERFNet
from utils.class_weights import get_class_weights
from utils.iou_meter import IoUMeter
from utils.logger import AutomatedLogger
from utils.misc import copy_object_sourcefile

device = torch.device('cuda')


def save_train_state(args, max_iou, enc: bool, epoch: int, is_best, model, optimizer, savedir, scheduler):
    if enc:
        path_checkpoint = savedir + '/checkpoint_enc.pth.tar'
        path_best = savedir + '/model_best_enc.pth.tar'
    else:
        path_checkpoint = savedir + '/checkpoint.pth.tar'
        path_best = savedir + '/model_best.pth.tar'
    save_checkpoint({
        'epoch':      epoch + 1,
        'arch':       str(model),
        'state_dict': model.state_dict(),
        'best_acc':   max_iou,
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
    }, is_best, path_checkpoint, path_best)

    if (enc):
        filename = f'{savedir}/model_encoder-{epoch:03}.pth'
        path_best = f'{savedir}/model_encoder_best.pth'
    else:
        filename = f'{savedir}/model-{epoch:03}.pth'
        path_best = f'{savedir}/model_best.pth'
    if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
        torch.save(model.state_dict(), filename)
        print(f'save: {filename} (epoch: {epoch})')
    if (is_best):
        torch.save(model.state_dict(), path_best)
        print(f'save: {path_best} (epoch: {epoch})')


def load_train_resume(enc, model, optimizer, savedir, scheduler):
    if enc:
        path_checkpoint = savedir + '/checkpoint_enc.pth.tar'
    else:
        path_checkpoint = savedir + '/checkpoint.pth.tar'
    assert os.path.exists(path_checkpoint)
    checkpoint = torch.load(path_checkpoint)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    max_iou = checkpoint['best_acc']
    scheduler.load_state_dict(checkpoint['scheduler'])
    print(f'checkpoint for epoch {checkpoint["epoch"]} loaded')
    return max_iou, start_epoch


def get_DataLoaders(args, enc):
    res = TRAIN_RESOLUTION[args.dataset + '_video']
    pair_transform = get_transform_seg(args.dataset, enc, augment=True, height=res.height, width=res.width)
    pair_transform_val = get_transform_seg(args.dataset, enc, augment=False, height=res.height, width=res.width)
    dataset_train = get_dataset(args.dataset, args.datalist, pair_transform, 'train', args.kth_frame)
    dataset_val = get_dataset(args.dataset, args.datalist, pair_transform_val, 'val')

    pin_mem = not args.no_pin_memory
    print(f'Pin memory: {pin_mem}')
    loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem)

    return loader_train, loader_val


def trainer(args, model, enc, epoch_callback_before=None):
    datasetMeta = get_dataset_metadata(args.dataset)
    res = TRAIN_RESOLUTION[args.dataset + '_video']
    weight = get_class_weights(args.dataset, args.datalist, enc, res.height, res.width, args.num_workers, args.batch_size, args.kth_frame).to(device)
    print(f'Weights are: {weight}')
    criterion = nn.CrossEntropyLoss(ignore_index=datasetMeta.IGNORE_INDEX, reduction='mean', weight=weight)

    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-4, betas=(0.9, 0.999))

    lam = lambda epoch: pow((1 - (epoch / args.num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    return train(args, model, enc, criterion, optimizer, scheduler, epoch_callback_before)


def train(args, model, enc, criterion, optimizer, scheduler, epoch_callback_before=None):
    max_iou = 0

    loader_train, loader_val = get_DataLoaders(args, enc)
    print(f'Dataset size train: {len(loader_train.dataset)}, val: {len(loader_val.dataset)}')

    savedir = args.savedir
    logger = AutomatedLogger(savedir, enc)

    start_epoch = 1
    if args.resume:
        max_iou, start_epoch = load_train_resume(enc, model, optimizer, savedir, scheduler)

    datasetMeta = get_dataset_metadata(args.dataset)

    for epoch in range(start_epoch, args.num_epochs + 1):
        if epoch_callback_before:
            epoch_callback_before(epoch, model)

        curr_lr = 0
        for param_group in optimizer.param_groups:
            curr_lr = float(param_group['lr'])
            print("CURRENT LR: ", param_group['lr'])

        iouTrain = IoUMeter(datasetMeta.NUM_CLASSES, datasetMeta.IGNORE_INDEX) if args.train_iou else None
        average_epoch_loss_train = do_train_epoch(args, model, optimizer, scheduler, criterion, loader_train, enc, epoch, iouTrain)

        iouVal = IoUMeter(datasetMeta.NUM_CLASSES, datasetMeta.IGNORE_INDEX) if args.val_iou else None
        average_epoch_loss_val = do_val_epoch(args, model, criterion, loader_val, enc, epoch, iouVal)

        logger.write(epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain.getIoU()[0] if iouTrain else 0, iouVal.getIoU()[0] if iouVal else 0, curr_lr, 0)

        if iouVal is None:
            curr_iou = -average_epoch_loss_val
        else:
            curr_iou = iouVal.getIoU()[0]

        is_best = curr_iou > max_iou
        max_iou = max(curr_iou, max_iou)
        save_train_state(args, max_iou, enc, epoch, is_best, model, optimizer, savedir, scheduler)
        if is_best:
            msg = f'max val iou={iouVal.getIoU()[0] * 100 if iouVal else 0:.2f}, in epoch {epoch}'
            if (not enc):
                with open(savedir + "/best.txt", "w") as f:
                    f.write(msg)
            else:
                with open(savedir + "/best_encoder.txt", "w") as f:
                    f.write(msg)
    return model


def do_train_epoch(args, model, optimizer, scheduler, criterion, loader: DataLoader, enc: bool, epoch: int, IoU: IoUMeter = None):
    print(f'======CURRENT EPOCH --- {epoch} --- TRAIN======')

    batch_losses = []
    batch_times = []

    if (IoU):
        IoU.reset()

    effective_to_simulated_batch_ratio = args.batch_size / args.gpu_batch_size

    model.train()
    for step, (images, labels) in enumerate(loader):
        start_time = time.time()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        avg_loss = 0

        for inputs, targets in zip(images.split(args.gpu_batch_size), labels.split(args.gpu_batch_size)):
            outputs = model(inputs, only_encode=enc)
            loss = criterion(outputs, targets[:, 0]) / effective_to_simulated_batch_ratio
            avg_loss += loss.item()
            loss.backward()

        optimizer.step()

        batch_losses.append(avg_loss)
        batch_times.append(time.time() - start_time)

        if (IoU):
            IoU.addBatch(outputs.max(1)[1].unsqueeze(1).detach(), targets.detach())

        if args.show_loss > 0 and step % args.show_loss == 0:
            avg = sum(batch_losses) / len(batch_losses)
            print(f'loss: {avg:0.4}, epoch: {epoch}, step: {step}, ',
                  f'Avg time/image: {sum(batch_times) / args.batch_size / len(batch_times):.4f}')

    average_epoch_loss = sum(batch_losses) / len(batch_losses)

    if (IoU):
        iou_val, iou_classes = IoU.getIoU()
        print(f'ep {epoch} MEAN IoU on train: {iou_val * 100:.2f}%')

    scheduler.step(epoch)

    return average_epoch_loss


def do_val_epoch(args, model, criterion, loader: DataLoader, enc: bool, epoch: int, IoU: IoUMeter = None):
    print(f'======CURRENT EPOCH --- {epoch} --- VALIDATE======')
    model.eval()
    batch_losses = []
    batch_times = []

    if (IoU):
        IoU.reset()

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, only_encode=enc)

            loss = criterion(outputs, labels[:, 0])
            batch_losses.append(loss.item())
            batch_times.append(time.time() - start_time)

            if (IoU):
                IoU.addBatch(outputs.max(1)[1].unsqueeze(1).detach(), labels.detach())

            if args.show_loss > 0 and step % args.show_loss == 0:
                avg = sum(batch_losses) / len(batch_losses)
                print(f'loss: {avg:0.4}, epoch: {epoch}, step: {step}, ',
                      f'Avg time/image: {sum(batch_times) / args.batch_size / len(batch_times):.4f}')

    average_epoch_loss = sum(batch_losses) / len(batch_losses)

    if (IoU):
        iou_val, iou_classes = IoU.getIoU()
        print(f'ep {epoch} MEAN IoU on val: {iou_val * 100:.2f}%')

    return average_epoch_loss


def save_checkpoint(state, is_best, path_checkpoint, path_best):
    torch.save(state, path_checkpoint)
    if is_best:
        torch.save(state, path_best)


def main(args):
    if not args.no_benchmark:
        cudnn.benchmark = True

    print(f'cudnn.enabled={cudnn.enabled}, cudnn.benchmark={cudnn.benchmark}')
    start_training = time.time()
    savedir = args.savedir

    if not exists(savedir):
        makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as f:
        f.write(str(args))

    datasetMeta = get_dataset_metadata(args.dataset)
    model = ERFNet(datasetMeta.NUM_CLASSES)
    copy_object_sourcefile(model, savedir)
    copyfile(__file__, savedir + '/' + basename(__file__))

    model = torch.nn.DataParallel(model).to(device)

    if args.weights:
        status = model.load_state_dict(torch.load(args.weights), strict=False)
        print(status)

    if (not args.decoder):
        print("-------TRAINING ENC-------")
        model = trainer(args, model, enc=True)
    print("-------TRAINING DEC-------")
    if (not args.weights):
        pretrainedEnc = next(model.children()).encoder
        model = ERFNet(datasetMeta.NUM_CLASSES, encoder=pretrainedEnc)
        model = torch.nn.DataParallel(model).to(device)

    trainer(args, model, enc=False)

    training_duration = time.time() - start_training
    minutes, seconds = divmod(int(training_duration), 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Training duration: {hours:02}:{minutes:02}:{seconds:02}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', help='file with initializing weights')

    parser.add_argument('--show-loss', type=int, default=200)
    parser.add_argument('--epochs-save', type=int, default=0, help='save every n epochs')
    parser.add_argument('--decoder', action='store_true', help='if specified, only train the decoder')

    parser.add_argument('--train-iou', action='store_true', default=False)
    parser.add_argument('--val-iou', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')

    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu-batch-size', type=int, default=None, help='the actual number of images processes simultaneously on GPU. If not specified, defaults to batch-size.')

    parser.add_argument('--savedir', default='resources/save/tests')

    parser.add_argument('--dataset', choices=['cityscapes', 'cityscapes_small', 'v_kitti'], default='v_kitti')
    parser.add_argument('--datalist', default='resources/dataset_lists/full.vkd')
    parser.add_argument('--kth-frame', type=int, help='In training, consider labels for every kth frame', default=4, required=False)

    parsed_args = parser.parse_args()

    if parsed_args.gpu_batch_size is None:
        parsed_args.gpu_batch_size = parsed_args.batch_size

    assert parsed_args.batch_size % parsed_args.gpu_batch_size == 0
    print(parsed_args)
    main(parsed_args)
