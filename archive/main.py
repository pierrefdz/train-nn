import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import submitit

from utils import progress_bar
    
parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', default='', type=str)
# parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--resume', '-r', action='store_true')

parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--dist_url', default='env://', type=str)

def main(args):
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Model
    print('==> Building model..')
    net = torchvision.models.resnet18(pretrained=False)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint')
        assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(checkpoint_path,'ckpt.pth'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # print(batch_idx, '/', len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch, best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                # print(batch_idx,'/', len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint')
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(state, os.path.join(checkpoint_path,'ckpt.pth'))
            best_acc = acc

    for epoch in range(start_epoch, start_epoch+500):
        train(epoch)
        test(epoch, best_acc)
        scheduler.step()


if __name__ == '__main__':


    args = parser.parse_args()

    # executor = submitit.AutoExecutor(folder=args.output_dir)
    # executor.update_parameters(
    #     timeout_min=2000, slurm_partition="devlab", gpus_per_node=4
    # )
    # _job = executor.submit(main, args)

    main(args)
