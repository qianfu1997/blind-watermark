# -*- coding: utf-8 -*-
# 原图像傅氏变换+水印图像随机编码->频域叠加+傅里叶逆变换
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
import pywt  # 离散小波变换
import random

ALPHA = 5  # 混杂强度
# 读取原始图像和水印图像
# numpy的傅氏变换
im = cv2.imread('1.jpg')/255  # 原始图像
mark = cv2.imread('lutos.jpg')/255  # 原始图像
imsize = im.size
im_height, im_width, im_channel = np.shape(im)
mark_height, mark_width = mark.shape[0], mark.shape[1]
# 原图像傅里叶变换
im_f = np.fft.fft2(im)  # 快速傅氏变换
im_f = np.fft.fftshift(im_f)  # 频移

# 随机编码
x = list(range(math.floor(im_height/2)))
y = list(range(im_width))
random.seed(im_height+im_width)
random.shuffle(x)
random.shuffle(y)
temp = np.zeros(im.shape)
for i in range(math.floor(im_height/2)):
    for j in range(im_width):
        if x[i] < mark_height and y[j] < mark_width:
            temp[i][j] = mark[x[i]][y[j]]
            temp[im_height-i-1][im_width-j-1] = temp[i][j]
            # 以上进行对称
fimg1 = 20*np.log(np.abs(im_f))
timf = im_f
im_f = im_f + ALPHA * temp
fimg2 = 20*np.log(np.abs(im_f))
# 逆变换
im_back = np.fft.ifftshift(im_f)
# im_back = im_f
im_back = np.abs(np.fft.ifft2(im_back))*255
# cv2.imwrite('test2.jpg', im_back,
#             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# test
wm = np.zeros(im.shape)
im_back_f = cv2.imread('test2.jpg')/255
im_back_f = np.fft.fft2(im_back_f)
im_back_f = np.fft.fftshift(im_back_f)
fimg3 = 20*np.log(np.abs(im_back_f))
tempMark = np.abs(im_back_f - timf)/ALPHA
for i in range(math.floor(im_height/2)):
    for j in range(im_width):
        wm[x[i]][y[j]] = tempMark[i][j]*255
cv2.imwrite('re1.jpg', wm, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imshow('original', np.uint8(im_back))
cv2.imshow('ff2', np.uint8(wm))
cv2.imshow('ff1', fimg3 - fimg1)
cv2.waitKey(0)


