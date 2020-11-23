import torch
import tqdm
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
import torch.nn.functional as F

def incremental_train_and_eval(epochs, tg_model, ref_model, free_model, ref_free_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 2.0
    beta = 0.25
    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
    for epoch in range(epochs):
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                    F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        if iteration == start_iteration:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
    return tg_model

