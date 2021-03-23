##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" The functions that compute the features """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils.misc import *
from utils.process_fp import process_inputs_fp

def compute_features(the_args, fusion_vars, tg_model, free_model, tg_feature_model, \
    is_start_iteration, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()
    tg_model.eval()
    if free_model is not None:
        free_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            if is_start_iteration:
                the_feature = tg_feature_model(inputs)
            else:
                the_feature = process_inputs_fp(the_args, fusion_vars, tg_model, free_model, inputs, feature_mode=True)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(the_feature.cpu())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
