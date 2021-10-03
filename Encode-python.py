# -*- coding: utf-8 -*-
# 原图像傅氏变换+水印图像随机编码->频域叠加+傅里叶逆变换
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
    parser.add_argument('--image', dest='im', required=True)
    parser.add_argument('--watermark', dest='mark', required=True)
    parser.add_argument('--result', dest='res', required=True)
    parser.add_argument('--alpha', dest='alpha', default=ALPHA)
    return parser


# main
def main():
    parser = build_parser()
    options = parser.parse_args()
    im = options.im
    mark = options.mark
    res = options.res
    alpha = float(options.alpha)
    if not os.path.isfile(im):
        parser.error("image %s does not exist." % im)
    if not os.path.isfile(mark):
        parser.error("watermark %s does not exist." % mark)
    # 调用加密函数,参数分别为源图像path，水印图像path，目标图像path，混杂强度
    encode(im, mark, res, alpha)


# encode方法
def encode(im_path, mark_path, res_path, alpha):
    # 读取源图像和水印图像
    im = cv2.imread(im_path)/255
    mark = cv2.imread(mark_path)/255
    im_height, im_width, im_channel = np.shape(im)
    mark_height, mark_width = mark.shape[0], mark.shape[1]
    # 源图像傅里叶变换 可换离散小波变换
    im_f = np.fft.fft2(im)
    im_f = np.fft.fftshift(im_f)
    # 水印图像编码
    # random
    x, y = list(range(math.floor(im_height/2))), list(range(im_width))
    random.seed(im_height+im_width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(im.shape)  # 与源图像等大小的模板，用于加上水印
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            if x[i] < mark_height and y[j] < mark_width:
                # 对称
                tmp[i][j] = mark[x[i]][y[j]]
                tmp[im_height-i-1][im_width-j-1] = tmp[i][j]
    # 混杂
    res_f = im_f + alpha * tmp
    # 逆变换
    res = np.fft.ifftshift(res_f)
    res = np.abs(np.fft.ifft2(res)) * 255  # 回乘255
    # 保存
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()







