#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

data_dir = 'data/imagenet/data/'

# Data loading code
traindir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(traindir, None)
classes = train_dataset.classes
print("the number of total classes: {}".format(len(classes)))

seed = 1993
np.random.seed(seed)
subset_num = 100
subset_classes = np.random.choice(classes, subset_num, replace=False)
print("the number of subset classes: {}".format(len(subset_classes)))
print(subset_classes)

des_root_dir = 'data/seed_{}_subset_{}_imagenet/data/'.format(seed, subset_num)
if not os.path.exists(des_root_dir):
    os.makedirs(des_root_dir)
phase_list = ['train', 'val']
for phase in phase_list:
    if not os.path.exists(os.path.join(des_root_dir, phase)):
        os.mkdir(os.path.join(des_root_dir, phase))
    for sc in subset_classes:
        src_dir = os.path.join(data_dir, phase, sc)
        des_dir = os.path.join(des_root_dir, phase, sc)
        cmd = "cp -r {} {}".format(src_dir, des_dir)
        print(cmd)
        os.system(cmd)

print("Generation finished.")