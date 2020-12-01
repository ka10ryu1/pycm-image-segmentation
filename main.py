#!/usr/bin/env python3
# -*-coding: utf-8 -*-
# pylint: disable=invalid-name,no-member
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from generate import MyDataset
from util import command, confusion_mat


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:3} [{:5}/{:5} ({:5.1%})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.mse_loss(output, target)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.9f}\n'.format(test_loss))


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # データローダー定義
    train_loader = DataLoader(MyDataset(100, img_num=1000), **train_kwargs)
    test_loader = DataLoader(MyDataset(101), **test_kwargs)
    # ネットワークモデル定義
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False
    ).to(device)

    # 学習させる場合
    if args.weight is None:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if not args.out_dir.exists():
            args.out_dir.mkdir(parents=True)

        torch.save(model.state_dict(), args.out_dir / 'unet_weight.pt')

    # 学習済みの重みを利用する場合
    else:
        state = torch.load(
            args.weight.as_posix(), map_location=lambda storage, loc: storage
        )
        model.load_state_dict(state)

    # PyCMで混同行列を計算
    confusion_mat(model, device, test_loader)


if __name__ == '__main__':
    main(command())
