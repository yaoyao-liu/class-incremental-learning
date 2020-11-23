from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
import torch.optim as optim
import torchvision
import argparse
import numpy as np
import os
import os.path as osp
import sys
import time
import math
import subprocess
try:
    import cPickle as pickle
except:
    import pickle

def savepickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not osp.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def tensor2im(input_image, imtype=np.uint8):
    mean = [0.5071,  0.4866,  0.4409]
    std = [0.2009,  0.1984,  0.2023]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().detach().float().numpy()
        if image_numpy.shape[0] == 1: 
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): 
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def process_mnemonics(X_protoset_cumuls, Y_protoset_cumuls, mnemonics, mnemonics_label, order_list):
    mnemonics_array_new = np.zeros(np.array(X_protoset_cumuls).shape)
    mnemonics_list = []
    mnemonics_label_list = []
    for idx in range(len(mnemonics)):
        this_mnemonics = []
        for sub_idx in range(len(mnemonics[idx])):
            processed_img = tensor2im(mnemonics[idx][sub_idx])
            mnemonics_array_new[idx][sub_idx] = processed_img
    
    diff = len(X_protoset_cumuls) - len(mnemonics_array_new)
    for idx in range(len(mnemonics_array_new)):
        X_protoset_cumuls[idx+diff] = mnemonics_array_new[idx]

    return X_protoset_cumuls