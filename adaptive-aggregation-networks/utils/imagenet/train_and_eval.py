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

from .utils_train import *

def train_and_eval(epochs, start_epoch, model, optimizer, lr_scheduler, \
        train_loader, val_loader, gpu=None):
    for epoch in range(start_epoch, epochs):
        lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(lr_scheduler.get_lr())

        train(train_loader, model, optimizer, epoch, gpu)
        validate(val_loader, model, gpu)

    return model

def train(train_loader, model, optimizer, epoch, gpu=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if gpu is not None:
            input = input.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
