import argparse
import json
import os
import shutil
import time
# import json
# import submitit
import numpy as np

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import utils

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Path parameters
    parser.add_argument('--data_dir', type=str, help='path to dataset')
    parser.add_argument('--train_data_dir', type=str, help='path to train dataset')
    parser.add_argument('--output_dir', type=str, default="")

    # Misc parameters
    parser.add_argument('--seed', default=0, type=int)

    # Dataloaders parameters
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--degrees', default=0, type=int, help='max degree in data-augmentation rotation (default: 0)')

    # Model parameters
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', "resnet50"])
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')

    # Optimization parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')

    # Distributed training parameters
    parser.add_argument('--debug_slurm', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--master_port', default=-1, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    # parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    # parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')

    return parser


def main(args):

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data loading code
    print("=> loading data from '{}'".format(args.data_dir))
    dir_train = os.path.join(args.data_dir, 'train') if args.train_data_dir is None else args.train_data_dir
    transform_train = transforms.Compose([
        transforms.RandomRotation(args.degrees),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = datasets.ImageFolder(dir_train, transform_train)

    dir_val = os.path.join(args.data_dir, 'val')
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_val = datasets.ImageFolder(dir_val, transform_val)
    print("=> data successfully loaded from '{}'".format(args.data_dir))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss()

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # optimizer = torch.optim.Adam(model_without_ddp.parameters(), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc1 = 0.0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.local_rank is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.local_rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.local_rank is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.local_rank)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args)
        val_stats = validate(val_loader, model, criterion, epoch, args)
        log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch
            }

        state_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

        if not args.distributed or (args.distributed and args.global_rank == 0):
            torch.save(state_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train(train_loader, model, criterion, optimizer, epoch, args):

    header = 'Train - Epoch: [{}/{}]'.format(epoch, args.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(metric_logger.log_every(train_loader, 10, header)):

        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_stats = {
            'loss': loss.item(),
            'acc1': acc1[0],
            'acc5': acc5[0],
        }
                
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
    
    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args):

    header = 'Val - Epoch: [{}/{}]'.format(epoch, args.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to val mode
    model.eval()

    for i, (images, target) in enumerate(metric_logger.log_every(val_loader, 10, header)):

        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        log_stats = {
            'loss': loss.item(),
            'acc1': acc1[0],
            'acc5': acc5[0],
        }
                
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
    
    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train Resnet18', parents=[get_args_parser()])
    args = parser.parse_args()

    # executor = submitit.AutoExecutor(folder=args.output_dir)
    # executor.update_parameters(
    #     timeout_min=2000, slurm_partition="devlab", gpus_per_node=4
    # )
    # _job = executor.submit(main, args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)