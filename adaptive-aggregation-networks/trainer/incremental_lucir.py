import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp

cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, balancedloader, iteration, start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list,lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_model.eval()

    num_old_classes = ref_model.fc.out_features
    handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
    handle_cur_features = b1_model.fc.register_forward_hook(get_cur_features)
    handle_old_scores_bs = b1_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
    handle_new_scores_bs = b1_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
    if iteration > start_iteration+1:
        ref_b2_model.eval()

    for epoch in range(epochs):
        b1_model.train()
        b2_model.train()

        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0

        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)

            tg_optimizer.zero_grad()

            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)

            if iteration == start_iteration+1:
                ref_outputs = ref_model(inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * lamda
            else:
                ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features_new.detach(), torch.ones(inputs.shape[0]).to(device)) * lamda
            loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==outputs.size())
            gt_index = torch.zeros(outputs_bs.size()).to(device)
            gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)
            max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
            hard_index = targets.lt(num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)
            if  hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
            else:
                loss3 = torch.zeros(1).to(device)
            loss = loss1 + loss2 + loss3

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train loss3: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))
        
        b1_model.eval()
        b2_model.eval()
        
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            fusion_optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            fusion_optimizer.step()

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
    handle_ref_features.remove()
    handle_cur_features.remove()
    handle_old_scores_bs.remove()
    handle_new_scores_bs.remove()
    return b1_model, b2_model

def incremental_train_and_eval_first_phase(the_args, epochs, b1_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, \
    lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        b1_model.train()

        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0

        tg_lr_scheduler.step()

        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = b1_model(inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            tg_optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))

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

