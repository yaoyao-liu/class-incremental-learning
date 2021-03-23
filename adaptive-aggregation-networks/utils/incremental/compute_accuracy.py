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
""" The functions that compute the accuracies """
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
from utils.imagenet.utils_dataset import merge_images_labels
from utils.process_fp import process_inputs_fp

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def compute_accuracy(the_args, fusion_vars, b1_model, b2_model, tg_feature_model, class_means, \
    X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=False, \
    fast_fc=None, scale=None, print_info=True, device=None, cifar=True, imagenet=False, \
    valdir=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b1_model.eval()
    tg_feature_model.eval()
    b1_model.eval()
    if b2_model is not None:
        b2_model.eval()
    fast_fc = 0.0
    correct = 0
    correct_icarl = 0
    correct_icarl_cosine = 0
    correct_icarl_cosine2 = 0
    correct_ncm = 0
    correct_maml = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            if is_start_iteration:
                outputs = b1_model(inputs)
            else:
                outputs, outputs_feature = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            if is_start_iteration:
                outputs_feature = np.squeeze(tg_feature_model(inputs))
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature.cpu(), 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            sqd_icarl_cosine = cdist(class_means[:,:,0].T, outputs_feature.cpu(), 'cosine')
            score_icarl_cosine = torch.from_numpy((-sqd_icarl_cosine).T).to(device)
            _, predicted_icarl_cosine = score_icarl_cosine.max(1)
            correct_icarl_cosine += predicted_icarl_cosine.eq(targets).sum().item()
            fast_weights = torch.from_numpy(np.float32(class_means[:,:,0].T)).to(device)
            sqd_icarl_cosine2 = F.linear(F.normalize(torch.squeeze(outputs_feature), p=2,dim=1), F.normalize(fast_weights, p=2, dim=1))
            score_icarl_cosine2 = sqd_icarl_cosine2
            _, predicted_icarl_cosine2 = score_icarl_cosine2.max(1)
            correct_icarl_cosine2 += predicted_icarl_cosine2.eq(targets).sum().item()
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature.cpu(), 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
    if print_info:
        print("  Current accuracy (FC)         :\t\t{:.2f} %".format(100.*correct/total))
        print("  Current accuracy (Proto)      :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  Current accuracy (Proto-UB)   :\t\t{:.2f} %".format(100.*correct_ncm/total))  
    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total
    return [cnn_acc, icarl_acc, ncm_acc], fast_fc
