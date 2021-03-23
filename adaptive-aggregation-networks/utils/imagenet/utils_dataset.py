##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Copied from: https://github.com/hshustc/CVPR19_Incremental_Learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Tools for ImageNet """
import argparse
import os
import shutil
import time
import numpy as np

def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert(len(images)==len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    
    return imgs
