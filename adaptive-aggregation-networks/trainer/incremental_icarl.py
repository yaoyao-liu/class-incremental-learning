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
""" Training code for iCaRL """
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

def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list,lamda, dist, K, lw_mr, balancedloader, T=None, beta=None, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features

    # If the 2nd branch reference is not None, set it to the evaluation mode
    if iteration > start_iteration+1:
        ref_b2_model.eval()

    for epoch in range(epochs):
        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()
        b2_model.train()

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
        fusion_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()

            # Forward the samples in the deep networks
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)

            if iteration == start_iteration+1:
                ref_outputs = ref_model(inputs)
            else:
                ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)
            # Loss 1: logits-level distillation loss
            loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
            # Loss 2: classification loss
            loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            # Sum up all looses
            loss = loss1 + loss2

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))
        
        # Update the aggregation weights
        b1_model.eval()
        b2_model.eval()
        
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            fusion_optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            fusion_optimizer.step()

        # Running the test for this epoch
        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register forward hook")
    return b1_model, b2_model
