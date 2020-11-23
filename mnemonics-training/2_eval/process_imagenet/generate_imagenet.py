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
from PIL import Image

src_root_dir = 'data/imagenet/data/'
des_root_dir = 'data/imagenet_resized_256/data/'
if not os.path.exists(des_root_dir):
    os.makedirs(des_root_dir)

phase_list = ['train', 'val']
for phase in phase_list:
    if not os.path.exists(os.path.join(des_root_dir, phase)):
        os.mkdir(os.path.join(des_root_dir, phase))
    data_dir = os.path.join(src_root_dir, phase)
    tg_dataset = datasets.ImageFolder(data_dir)
    for cls_name in tg_dataset.classes:
        if not os.path.exists(os.path.join(des_root_dir, phase, cls_name)):
            os.mkdir(os.path.join(des_root_dir, phase, cls_name))
    cnt = 0
    for item in tg_dataset.imgs:
        img_path = item[0]
        img = Image.open(img_path)
        img = img.convert('RGB')
        save_path = img_path.replace('imagenet', 'imagenet_resized_256')
        resized_img = img.resize((256,256), Image.BILINEAR)
        resized_img.save(save_path)
        cnt = cnt+1
        if cnt % 1000 == 0:
            print(cnt, save_path)

print("Generation finished.")
