
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn.functional as F
import math
import torch.nn as nn
from copy import deepcopy
from math import exp

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def compute_loss(input, target_xyz, rec_xyz, rendered):
    loss = torch.sum(torch.abs(input - rendered) + (
        1.5 * torch.abs(target_xyz - rec_xyz)))/input.size(0)
    return loss

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    # return torch.unsqueeze(torch.from_numpy(image), dim=0)
    return torch.from_numpy(image)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def imshow(img, xyz_out=None, srgb_out=None, task=None):
    """ displays images """

    if task.lower() == 'srgb-2-xyz-2-srgb':
        if xyz_out is None:
            raise Exception('XYZ image is not given')
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')
        ax[2].set_title('re-rendered')
        ax[2].imshow(from_bgr2rgb(srgb_out))
        ax[2].axis('off')

    if task.lower() == 'srgb-2-xyz':
        if xyz_out is None:
            raise Exception('XYZ image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')

    if task.lower() == 'xyz-2-srgb':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('re-rendered')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    if task.lower() == 'pp':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    plt.xticks([]), plt.yticks([])
    plt.show()


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    if im[0].dtype == 'uint8':
        max_value = 255
    elif im[0].dtype == 'uint16':
        max_value = 65535
    return im.astype('float') / max_value


def denorm(img, max_value):
    img = img * float(max_value)
    return img

def norm_img(img, max_value):
    img = img / float(max_value)
    return img


def calc_para(net):
    num_params = 0
    f_params = 0
    m_params = 0
    l_params = 0
    stage = 1
    total_str = 'The number of parameters for each sub-block:\n'

    for param in net.parameters():
        num_params += param.numel()

# 计算网络各部分参数量
    for body in net.named_children():
        res_params = 0
        res_str = []
        for param in body[1].parameters():
            res_params += param.numel()
        res_str = '[{:s}] parameters: {}\n'.format(body[0], res_params)
        total_str = total_str + res_str
        if stage == 1:
            f_params = f_params + res_params
            # if body[0] == 'base_detail':
            #     stage = 2
        elif stage == 2:
            m_params = m_params + res_params
            # if body[0] == 'conv2d':
            #     stage = 3
        elif stage == 3:
            l_params = l_params + res_params
        if 'anchor' in body[0]:     stage += 1

    total_str = total_str + '[total] parameters: {}\n\n'.format(num_params) + \
                '[first_net]\tparameters: {:.4f} M\n'.format(f_params/1e6) + \
                '[middle_net]parameters: {:.4f} M\n'.format(m_params/1e6) + \
                '[last_net]\tparameters: {:.4f} M\n'.format(l_params/1e6) + \
                '[total_net]\tparameters: {:.4f} M\n'.format(num_params/1e6) + \
                '***'
    return total_str

