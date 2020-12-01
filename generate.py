#!/usr/bin/env python3
# -*-coding: utf-8 -*-
# pylint: disable=invalid-name
import string
from PIL import Image

import cv2
import numpy as np
from numpy.random import randint as rndint
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, seed, resize=128, img_num=100):
        self.ig = ImgGenerator(400, 400, seed)
        self.img_num = img_num
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        self.transform_y = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # 画像生成呼び出し
        x, y = read_imgs(self.ig)
        # transformsがPIL形式じゃないと使えないので、transformする前に変換
        x = self.transform_x(Image.fromarray(x))
        y = self.transform_y(Image.fromarray(y))
        return x, y

    def __len__(self):
        return self.img_num


class ImgGenerator(object):

    def __init__(self, h=400, w=600, seed=-1, white=False):
        self._h = h
        self._w = w
        if seed > 0:
            np.random.seed(seed)

        self.reset(white)

    @property
    def img(self):
        return self._img.copy()

    def _reset(self):
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def _is_mono(self, img):
        return len(img.shape) == 2 or img.shape[-1] != 3

    def _is_color(self, img):
        return not self._is_mono(img)

    def _gray(self, img):
        if self._is_color(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _bgr(self, img):
        if self._is_mono(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def _rgb(self, img):
        flg = cv2.COLOR_GRAY2BGR if self._is_mono(img) else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(img, flg)

    def _rnd_bgr(self, _min, _max):
        if _min > _max:
            buf = _max
            _max = _min
            _min = buf

        _min = np.max([_min, 0])
        _max = np.min([_max, 255])
        return (rndint(_min, _max), rndint(_min, _max), rndint(_min, _max))

    def reset(self, white=False):
        self._img = self._reset()
        if white:
            self._img.fill(255)

    def to_gray(self):
        self._img = self._gray(self._img)
        return self

    def to_bgr(self):
        self._img = self._bgr(self._img)
        return self

    def to_rgb(self):
        self._img = self._rgb(self._img)
        return self

    def to_binary(self, th=125, max_val=255, flg=cv2.THRESH_BINARY):
        img = self._gray(self.img)
        self._img = cv2.threshold(img, th, max_val, flg)[1]
        return self

    def bitwise_and(self, mask):
        mask = cv2.bitwise_not(mask)
        self._img = cv2.merge([
            cv2.bitwise_and(img, mask)
            for img in cv2.split(self._img)
        ])
        return self

    def add_pattern(self, color_range=(50, 255)):
        self._img = self._bgr(self._img)
        bk = self._img
        h, w = bk.shape[:2]
        for i in range(rndint(3, 15)):
            pt1 = (rndint(0, w), rndint(0, h))
            _pt = (rndint(i * 3, w // 2), rndint(i * 3, h // 2))
            pt2 = (pt1[0] + _pt[0], pt1[1] + _pt[1])
            color = self._rnd_bgr(*color_range)
            bk = cv2.rectangle(bk, pt1, pt2, color, -1)

        for i in range(rndint(3, 15)):
            size = (rndint(0, w), rndint(0, h))
            radius = rndint(5, 15) * i
            color = self._rnd_bgr(*color_range)
            bk = cv2.circle(bk, size, radius, color, -1)

        self._img = bk
        return self

    def add_str(
        self, color_range=(50, 255),
        font=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_8
    ):
        ascii_str = list(string.ascii_letters + string.octdigits)
        bk = self._img
        h, w = bk.shape[:2]
        for i in np.random.choice(ascii_str, np.random.randint(5, 10)):
            pt = (rndint(10, w), rndint(10, h))
            th = rndint(2, 5)
            size = rndint(2, 5)
            color = self._rnd_bgr(*color_range)
            cv2.putText(bk, i, pt, font, size, color, th, line_type)

        return self


def read_imgs(ig):
    # 文字を追加
    img = ig.add_str().img
    # 画像を二値化して正解画像を作成
    target_img = ig.to_binary(10).img
    # 背景画像生成のためにリセット
    ig.reset()
    # 背景を追加して、文字を入れる分を除去する
    bg = ig.add_pattern().bitwise_and(target_img).img
    # 背景画像と文字画像で入出力画像を生成
    input_img = img + bg
    # 次の処理のためにリセット
    ig.reset()
    return input_img, target_img


def sample(wait_time=0):
    import cv2
    import numpy as np

    ig = ImgGenerator(300, 300, 1)
    input_imgs = list()
    target_imgs = list()
    for _ in range(5):
        img1, img2 = read_imgs(ig)
        input_imgs.append(img1)
        target_imgs.append(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR))

    img = np.vstack([
        np.hstack(input_imgs),
        np.hstack(target_imgs)
    ])
    cv2.imshow('imgs', img)
    cv2.waitKey(wait_time)
    cv2.imwrite('sample.png', img)


if __name__ == '__main__':
    sample()
