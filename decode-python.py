# -*- coding: utf-8 -*-
# 水印图像傅里叶变换，图像傅里叶电环相减，解码后得到水印
import cv2
import numpy as np
import random
import math
import os
from argparse import ArgumentParser

# default混杂强度
ALPHA = 15


def build_parser():
    parser = ArgumentParser()
    # 原始图像+加密图像+解密图像地址+混杂强度
    parser.add_argument('--original', dest='ori', required=True)
    parser.add_argument('--image', dest='im', required=True)
    parser.add_argument('--result', dest='res', required=True)
    parser.add_argument('--alpha', dest='alpha', default=ALPHA)
    return parser


# main
def main():
    parser = build_parser()
    options = parser.parse_args()
    ori = options.ori
    im = options.im
    res = options.res
    alpha = options.alpha
    if not os.path.isfile(im):
        parser.error("image %s does not exist." % im)
    if not os.path.isfile(ori):
        parser.error("original %s does not exist." % ori)
    # 调用加密函数,参数分别为源图像path，水印图像path，提出水印path，混杂强度
    decode(ori, im, res, alpha)


# decode
def decode(ori_path, im_path, res_path, alpha):
    ori = cv2.imread(ori_path)/255
    im = cv2.imread(im_path)/255
    im_height, im_width, im_channel = np.shape(ori)
    # 源图像与水印图像傅里叶变换
    ori_f = np.fft.fft2(ori)
    ori_f = np.fft.fftshift(ori_f)
    im_f = np.fft.fft2(im)
    im_f = np.fft.fftshift(im_f)
    mark = np.abs((im_f - ori_f) / alpha)
    res = np.zeros(ori.shape)

    # 获取随机种子
    x, y = list(range(math.floor(im_height/2))), list(range(im_width))
    random.seed(im_height+im_width)
    random.shuffle(x)
    random.shuffle(y)
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            res[x[i]][y[j]] = mark[i][j]*255
            res[im_height-i-1][im_width-j-1] = res[i][j]
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()






