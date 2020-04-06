import itertools, imageio, torch, random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets
# from scipy.misc import imresize
from PIL import Image 
from torch.autograd import Variable
import os

class ToRGB(object):
    """
    Convert the given PIL Image to RGB.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be RGBed.

        Returns:
            PIL Image: RGB image.
        """
        return img.convert("RGB")

class RatioedResize(object):
    """
    Resize the given PIL Image while keeping ratio
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        ratio = h *1.0 / w
        if ratio > 1:
            h = self.size
            w = int(h*1.0/ratio)
        else:
            w = self.size
            h = int(w * ratio)
        result = img.resize((h, w), Image.BICUBIC)

        return result

class RGBToBGR(object):
    """
    Convert the given Tensor Image in RGB to BGR.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): RGB Image to be BGRed.

        Returns:
            PIL Image: RGB image.
        """
        return img[[2, 1, 0], :, :]

class Zero(object):
    """
    Zeroing the given Tensor Image.
    """
    def __call__(self, img):
        return -1 + 2 * img

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=os.cpu_count())

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
