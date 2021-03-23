##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Training code for the 0-th phase """
import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
import torch.nn.functional as F

def incremental_train_and_eval_zeroth_phase(the_args, epochs, b1_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, \
    lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        # Set the 1st branch model to the training mode
        b1_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        # Set the counters to zeros
        correct = 0
        total = 0

        # Learning rate decay
        tg_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)
            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()
            # Forward the samples in the deep networks
            outputs = b1_model(inputs)
            # Compute classification loss
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()
            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))

        # Running the test for this epoch
        b1_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = b1_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    return b1_model
