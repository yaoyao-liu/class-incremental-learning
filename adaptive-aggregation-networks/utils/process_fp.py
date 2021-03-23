##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Using the aggregation weights to compute the feature maps from two branches """
import torch
import torch.nn as nn
from utils.misc import *

def process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs, feature_mode=False):

    # The 1st level
    if the_args.dataset == 'cifar100':
        b1_model_group1 = [b1_model.conv1, b1_model.bn1, b1_model.relu, b1_model.layer1]
        b2_model_group1 = [b2_model.conv1, b2_model.bn1, b2_model.relu, b2_model.layer1]
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        b1_model_group1 = [b1_model.conv1, b1_model.bn1, b1_model.relu, b1_model.maxpool, b1_model.layer1]
        b2_model_group1 = [b2_model.conv1, b2_model.bn1, b2_model.relu, b2_model.maxpool, b2_model.layer1]
    else:
        raise ValueError('Please set correct dataset.')
    b1_model_group1 = nn.Sequential(*b1_model_group1)
    b1_fp1 = b1_model_group1(inputs)
    b2_model_group1 = nn.Sequential(*b2_model_group1)
    b2_fp1 = b2_model_group1(inputs)
    fp1 = fusion_vars[0]*b1_fp1+(1-fusion_vars[0])*b2_fp1

    # The 2nd level
    b1_model_group2 = b1_model.layer2
    b1_fp2 = b1_model_group2(fp1)
    b2_model_group2 = b2_model.layer2
    b2_fp2 = b2_model_group2(fp1)
    fp2 = fusion_vars[1]*b1_fp2+(1-fusion_vars[1])*b2_fp2

    # The 3rd level
    if the_args.dataset == 'cifar100':
        b1_model_group3 = [b1_model.layer3, b1_model.avgpool]
        b2_model_group3 = [b2_model.layer3, b2_model.avgpool]
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        b1_model_group3 = b1_model.layer3
        b2_model_group3 = b2_model.layer3
    else:
        raise ValueError('Please set correct dataset.')
    b1_model_group3 = nn.Sequential(*b1_model_group3)
    b1_fp3 = b1_model_group3(fp2)
    b2_model_group3 = nn.Sequential(*b2_model_group3)
    b2_fp3 = b2_model_group3(fp2)
    fp3 = fusion_vars[2]*b1_fp3+(1-fusion_vars[2])*b2_fp3

    if the_args.dataset == 'cifar100': 
        fp_final = fp3.view(fp3.size(0), -1)
    elif the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
        # The 4th level
        b1_model_group4 = [b1_model.layer4, b1_model.avgpool]
        b1_model_group4 = nn.Sequential(*b1_model_group4)
        b1_fp4 = b1_model_group4(fp3)
        b2_model_group4 = [b2_model.layer4, b2_model.avgpool]
        b2_model_group4 = nn.Sequential(*b2_model_group4)
        b2_fp4 = b2_model_group4(fp3)
        fp4 = fusion_vars[3]*b1_fp4+(1-fusion_vars[3])*b2_fp4
        fp_final = fp4.view(fp4.size(0), -1)
    else:
        raise ValueError('Please set correct dataset.')
    if feature_mode:
        return fp_final
    else:
        outputs = b1_model.fc(fp_final)
        return outputs, fp_final
