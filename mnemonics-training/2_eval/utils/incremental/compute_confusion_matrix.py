#!/usr/bin/env python
# coding=utf-8
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

def compute_confusion_matrix(tg_model, tg_feature_model, class_means, evalloader, print_info=False, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    num_classes = tg_model.fc.out_features
    cm = np.zeros((3, num_classes, num_classes))
    all_targets = []
    all_predicted = []
    all_predicted_icarl = []
    all_predicted_ncm = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets)

            outputs = tg_model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted)



    cm[0, :, :] = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))

    if print_info:
        print("  top 1 accuracy            :\t\t{:.2f} %".format( 100.*correct/total ))
    return cm
