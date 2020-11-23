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
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_resnet as modified_resnet
import models.modified_resnetmtl as modified_resnetmtl
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_features import compute_features
from utils.incremental.compute_accuracy import compute_accuracy
from utils.incremental.compute_confusion_matrix import compute_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self, the_args):
        self.args = the_args
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + self.args.dataset + '_nfg' + str(self.args.nb_cl_fg) + '_ncls' + str(self.args.nb_cl) + \
        '_nproto' + str(self.args.nb_protos) 
        if self.args.use_mtl:
            self.save_path += '_mtl'
        if self.args.add_str is not None:
            self.save_path += self.args.add_str          
        self.save_path += '_' + str(self.args.ckpt_label)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.args.dataset == 'cifar100':
            self.transform_train = transforms.Compose([ \
                transforms.RandomCrop(32, padding=4), \
                transforms.RandomHorizontalFlip(), \
                transforms.ToTensor(), \
                transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            self.transform_test = transforms.Compose([ \
                transforms.ToTensor(), \
                transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_test)

            self.network = modified_resnet_cifar.resnet32
            self.network_mtl = modified_resnetmtl_cifar.resnetmtl32
            self.lr_strat = [int(self.args.epochs*0.5), int(self.args.epochs*0.75)]
            self.dictionary_size = 500

        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            traindir = os.path.join(self.args.data_dir, 'train')
            valdir = os.path.join(self.args.data_dir, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trainset = datasets.ImageFolder(traindir, \
                transforms.Compose([transforms.RandomResizedCrop(224), \
                transforms.RandomHorizontalFlip(), \
                transforms.ToTensor(), normalize,]))
            self.testset =  datasets.ImageFolder(valdir, \
                transforms.Compose([transforms.Resize(256), \
                transforms.CenterCrop(224), \
                transforms.ToTensor(), normalize, ]))
            self.evalset =  datasets.ImageFolder(valdir, \
                transforms.Compose([transforms.Resize(256), \
                transforms.CenterCrop(224), \
                transforms.ToTensor(), normalize,]))

            self.network = modified_resnet.resnet18
            self.network_mtl = modified_resnetmtl.resnetmtl18
            self.lr_strat = [30, 60]
            self.dictionary_size = 1500

        else:
            raise ValueError('Please set correct dataset.')


    def eval(self):
        self.train_writer = SummaryWriter(comment=self.save_path)
        dictionary_size = self.dictionary_size
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))

        if self.args.dataset == 'cifar100':
            X_train_total = np.array(self.trainset.train_data)
            Y_train_total = np.array(self.trainset.train_labels)
            X_valid_total = np.array(self.testset.test_data)
            Y_valid_total = np.array(self.testset.test_labels)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            X_train_total, Y_train_total = split_images_labels(self.trainset.imgs)
            X_valid_total, Y_valid_total = split_images_labels(self.testset.imgs)
        else:
            raise ValueError('Please set correct dataset.')

        for iteration_total in range(self.args.nb_runs):
            order_name = osp.join(self.save_path, \
                "seed_{}_{}_order_run_{}.pkl".format(self.args.random_seed, self.args.dataset, iteration_total))
            print("Order name:{}".format(order_name))

            if osp.exists(order_name):
                print("Loading orders")
                order = utils.misc.unpickle(order_name)
            else:
                print("Generating orders")
                order = np.arange(self.args.num_classes)
                np.random.shuffle(order)
                utils.misc.savepickle(order, order_name)
            order_list = list(order)
            print(order_list)

        X_valid_cumuls    = []
        X_protoset_cumuls = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []

        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            if iteration == start_iter:
                last_iter = 0
                tg_model = self.network(num_classes=self.args.nb_cl_fg)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                ref_model = None
            elif iteration == start_iter+1:
                last_iter = iteration
                ref_model = copy.deepcopy(tg_model)
                if self.args.use_mtl:
                    tg_model = self.network_mtl(num_classes=self.args.nb_cl_fg)
                else:
                    tg_model = self.network(num_classes=self.args.nb_cl_fg)
                ref_dict = ref_model.state_dict()
                tg_dict = tg_model.state_dict()
                tg_dict.update(ref_dict)
                tg_model.load_state_dict(tg_dict)
                tg_model.to(self.device)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("in_features:", in_features, "out_features:", out_features)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
                new_fc.fc1.weight.data = tg_model.fc.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = out_features*1.0 / self.args.nb_cl
            else:
                last_iter = iteration
                ref_model = copy.deepcopy(tg_model)
                in_features = tg_model.fc.in_features
                out_features1 = tg_model.fc.fc1.out_features
                out_features2 = tg_model.fc.fc2.out_features
                print("in_features:", in_features, "out_features1:", out_features1, "out_features2:", out_features2)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
                new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = (out_features1+out_features2)*1.0 / (self.args.nb_cl)

            actual_cl = order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)]
            indices_train_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_train_total])
            indices_test_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_valid_total])

            X_valid = X_valid_total[indices_test_10]
            X_valid_cumuls.append(X_valid)
            X_valid_cumul = np.concatenate(X_valid_cumuls)

            Y_valid = Y_valid_total[indices_test_10]
            Y_valid_cumuls.append(Y_valid)
            Y_valid_cumul = np.concatenate(Y_valid_cumuls)

            if iteration == start_iter:
                X_valid_ori = X_valid
                Y_valid_ori = Y_valid

            ckp_name = osp.join(self.save_path, 'run_{}_iteration_{}_model.pth'.format(iteration_total, iteration))

            print('ckp_name', ckp_name)
            print("[*] Loading models from checkpoint")
            tg_model = torch.load(ckp_name)
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])

            if self.args.dataset == 'cifar100':
                map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                print('Computing accuracy on the original batch of classes...')
                self.evalset.test_data = X_valid_ori.astype('uint8')
                self.evalset.test_labels = map_Y_valid_ori
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                ori_acc = compute_accuracy(tg_model, tg_feature_model, evalloader)
                top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
                self.train_writer.add_scalar('ori_acc/cnn', float(ori_acc), iteration)
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
                print('Computing cumulative accuracy...')
                self.evalset.test_data = X_valid_cumul.astype('uint8')
                self.evalset.test_labels = map_Y_valid_cumul
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)        
                cumul_acc = compute_accuracy(tg_model, tg_feature_model, evalloader)
                top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
                self.train_writer.add_scalar('cumul_acc/cnn', float(cumul_acc), iteration)
            elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
                map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                print('Computing accuracy on the original batch of classes...')
                current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                ori_acc = compute_accuracy(tg_model, tg_feature_model, evalloader)
                top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
                self.train_writer.add_scalar('ori_acc/cnn', float(ori_acc), iteration)
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
                print('Computing cumulative accuracy...')
                current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers, pin_memory=True)        
                cumul_acc = compute_accuracy(tg_model, tg_feature_model, evalloader)
                top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
                self.train_writer.add_scalar('cumul_acc/cnn', float(cumul_acc), iteration)
            else:
                raise ValueError('Please set correct dataset.')

        self.train_writer.close()
