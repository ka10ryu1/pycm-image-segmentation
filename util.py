#!/usr/bin/env python3
# -*-coding: utf-8 -*-
# pylint: disable=invalid-name,no-member
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from pycm import ConfusionMatrix


def _thresh(img, th=125, max_val=255, flg=cv2.THRESH_BINARY):
    kwargs = {'thresh': th, 'maxval': max_val, 'type': flg}
    return cv2.threshold(img, **kwargs)[1].astype(np.uint8)


def _to_img(tensor, min_val=0, max_val=255, dtype=np.uint8, binary=True):
    if isinstance(tensor, np.ndarray):
        return tensor

    img = np.transpose(tensor.numpy(), (1, 2, 0))
    img = img * max_val
    img = np.clip(img, min_val, max_val)
    img = img.astype(dtype)
    if binary:
        img = _thresh(img)

    return img


def _to_imgs(tensors, **kwargs):
    return [_to_img(tensor, **kwargs) for tensor in tensors]


def _none_chk(val):
    return 0 if val == 'None' else val


def confusion_no_show(model, device, test_loader, fmt='6.1%', add_out=False):
    if add_out:
        return confusion_no_show3(model, device, test_loader, fmt)

    model.eval()
    img = []
    print(' TPR    PPV    G      J      F1     AUPR')
    with torch.no_grad():
        # バッチサイズ毎に画像を読み込み
        for data, target in test_loader:
            tgt = _to_imgs(target)
            src = _to_imgs(data, binary=False)
            data, target = data.to(device), target.to(device)
            out = _to_imgs(model(data).cpu())
            # 画像一枚ずつ処理
            for x, y, z in zip(tgt, out, src):
                # 画像の可視化に必要な処理
                b = np.zeros_like(x)
                tp = cv2.bitwise_and(x, y)
                ntp = cv2.bitwise_not(tp)
                fp = cv2.bitwise_and(y, ntp)
                fn = cv2.bitwise_and(x, ntp)
                dst = cv2.merge([b, tp + fp, fn + fp])
                img.append(cv2.resize(np.vstack([z, dst]), (300, 600)))

                # PyCM用の処理
                x = x.reshape(-1) // 255
                y = y.reshape(-1) // 255
                cm = ConfusionMatrix(actual_vector=x, predict_vector=y)

                print(
                    f'{_none_chk(cm.TPR[1]):{fmt}},{_none_chk(cm.PPV[1]):{fmt}},' +
                    f'{_none_chk(cm.G[1]):{fmt}},{_none_chk(cm.J[1]):{fmt}},' +
                    f'{_none_chk(cm.F1[1]):{fmt}},{_none_chk(cm.AUPR[1]):{fmt}}'
                )

    return np.hstack(img)


def add_border(img):
    return cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def confusion_no_show3(model, device, test_loader, fmt='6.1%'):
    model.eval()
    img = []
    print(' TPR    PPV    G      J      F1     AUPR')
    with torch.no_grad():
        # バッチサイズ毎に画像を読み込み
        for data, target in test_loader:
            tgt = _to_imgs(target)
            src = _to_imgs(data, binary=False)
            data, target = data.to(device), target.to(device)
            out = _to_imgs(model(data).cpu())
            # 画像一枚ずつ処理
            for x, y, z in zip(tgt, out, src):
                # 画像の可視化に必要な処理
                b = np.zeros_like(x)
                tp = cv2.bitwise_and(x, y)
                ntp = cv2.bitwise_not(tp)
                fp = cv2.bitwise_and(y, ntp)
                fn = cv2.bitwise_and(x, ntp)
                dst = cv2.merge([b, tp + fp, fn + fp])
                y2 = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                img.append(
                    cv2.resize(
                        np.vstack([
                            add_border(z),
                            add_border(y2),
                            add_border(dst)
                        ]), (300, 900)
                    )
                )

                # PyCM用の処理
                x = x.reshape(-1) // 255
                y = y.reshape(-1) // 255
                cm = ConfusionMatrix(actual_vector=x, predict_vector=y)

                print(
                    f'{_none_chk(cm.TPR[1]):{fmt}},{_none_chk(cm.PPV[1]):{fmt}},' +
                    f'{_none_chk(cm.G[1]):{fmt}},{_none_chk(cm.J[1]):{fmt}},' +
                    f'{_none_chk(cm.F1[1]):{fmt}},{_none_chk(cm.AUPR[1]):{fmt}}'
                )

    return np.hstack(img)


def confusion_mat(model, device, test_loader, fmt='6.1%'):
    model.eval()
    # input_imgs = list()
    # target_imgs = list()
    print(' TPR    PPV    G      J      F1     AUPR')
    with torch.no_grad():
        # バッチサイズ毎に画像を読み込み
        for data, target in test_loader:
            tgt = _to_imgs(target)
            src = _to_imgs(data, binary=False)
            data, target = data.to(device), target.to(device)
            out = _to_imgs(model(data).cpu())
            # 画像一枚ずつ処理
            for x, y, z in zip(tgt, out, src):
                # 画像の可視化に必要な処理
                b = np.zeros_like(x)
                tp = cv2.bitwise_and(x, y)
                ntp = cv2.bitwise_not(tp)
                fp = cv2.bitwise_and(y, ntp)
                fn = cv2.bitwise_and(x, ntp)
                dst = cv2.merge([b, tp + fp, fn + fp])

                cv2.imshow('test', cv2.resize(np.vstack([z, dst]), (300, 600)))
                # input_imgs.append(z)
                # target_imgs.append(dst)
                if cv2.waitKey(20) == ord('q'):
                    return 0

                # PyCM用の処理
                x = x.reshape(-1) // 255
                y = y.reshape(-1) // 255
                cm = ConfusionMatrix(actual_vector=x, predict_vector=y)

                print(
                    f'{_none_chk(cm.TPR[1]):{fmt}},{_none_chk(cm.PPV[1]):{fmt}},' +
                    f'{_none_chk(cm.G[1]):{fmt}},{_none_chk(cm.J[1]):{fmt}},' +
                    f'{_none_chk(cm.F1[1]):{fmt}},{_none_chk(cm.AUPR[1]):{fmt}}'
                )

    # img = np.vstack([np.hstack(input_imgs[:5]),np.hstack(target_imgs[:5])])
    # cv2.imwrite(
    #     'result.png',
    #     cv2.resize(img, (1500, 600), interpolation=cv2.INTER_CUBIC)
    # )


def command():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size', type=int, default=16, metavar='N',
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 14)'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0, metavar='LR',
        help='learning rate (default: 1.0)'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.7, metavar='M',
        help='Learning rate step gamma (default: 0.7)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--dry-run', action='store_true', default=False,
        help='quickly check a single pass'
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--weight', metavar='PATH', type=Path, default=None,
        help='use initial weight [default: None]'
    )
    parser.add_argument(
        '--out_dir', metavar='PATH', type=Path, default=Path.cwd() / 'out',
        help='result output dir [default: ./out]'
    )

    return parser.parse_args()
