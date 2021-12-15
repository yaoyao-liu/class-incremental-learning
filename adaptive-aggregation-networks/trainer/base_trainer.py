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
""" Class-incremental learning base trainer. """
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
from utils.misc import process_mnemonics
import warnings
warnings.filterwarnings('ignore')

class BaseTrainer(object):
    """The class that contains the code for base trainer class.
    This file only contains the related functions used in the training process.
    If you hope to view the overall training process, you may find it in the file named trainer.py in the same folder.
    """
    def __init__(self, the_args):
        """The function to initialize this class.
        Args:
          the_args: all inputted parameter.
        """
        self.args = the_args
        self.set_save_path()
        self.set_cuda_device()
        self.set_dataset_variables()

    def set_save_path(self):
        """The function to set the saving path."""
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + self.args.dataset + \
            '_nfg' + str(self.args.nb_cl_fg) + \
            '_ncls' + str(self.args.nb_cl) + \
            '_nproto' + str(self.args.nb_protos) + \
            '_' + self.args.baseline + \
            '_' + self.args.branch_mode + \
            '_b1' + self.args.branch_1

        if self.args.branch_mode == 'dual':
            self.save_path += '_b2' + self.args.branch_2

        if self.args.dynamic_budget:
            self.save_path += '_dynamic'
        else:
            self.save_path += '_fixed'  

        self.save_path += '_' + str(self.args.ckpt_label)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path) 

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       

    def set_dataset_variables(self):
        """The function to set the dataset parameters."""
        if self.args.dataset == 'cifar100':
            # Set CIFAR-100
            # Set the pre-processing steps for training set
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            # Set the pre-processing steps for test set
            self.transform_test = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            # Initial the dataloader
            self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_test)
            self.balancedset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_train)
            # Set the network architecture
            self.network = modified_resnet_cifar.resnet32
            self.network_mtl = modified_resnetmtl_cifar.resnetmtl32
            # Set the learning rate decay parameters
            self.lr_strat = [int(self.args.epochs*0.5), int(self.args.epochs*0.75)]
            # Set the dictionary size
            self.dictionary_size = 500

        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Set imagenet-subset and imagenet
            # Set the data directories
            traindir = os.path.join(self.args.data_dir, 'train')
            valdir = os.path.join(self.args.data_dir, 'val')
            # Set the dataloaders
            train_transforms = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=63 / 255)]
            test_transforms = [transforms.Resize(256),transforms.CenterCrop(224)]
            common_transforms = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            train_trsf = transforms.Compose([*train_transforms, *common_transforms])
            test_trsf = transforms.Compose([*test_transforms, *common_transforms])
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trainset = datasets.ImageFolder(traindir, train_trsf)
            self.testset =  datasets.ImageFolder(valdir, test_trsf)
            self.evalset =  datasets.ImageFolder(valdir, test_trsf)
            self.balancedset =  datasets.ImageFolder(traindir, train_trsf)

            # Set the network architecture
            if self.args.imgnet_backbone == 'resnet18':
                self.network = modified_resnet.resnet18
                self.network_mtl = modified_resnetmtl.resnetmtl18
            elif self.args.imgnet_backbone == 'resnet34':
                self.network = modified_resnet.resnet34
                self.network_mtl = modified_resnetmtl.resnetmtl34
            else:
                raise ValueError('Please set the correct backbone.')
            # Set the learning rate decay parameters
            self.lr_strat = [int(self.args.epochs*0.333), int(self.args.epochs*0.667)]
            # Set the dictionary size
            self.dictionary_size = 1500
        else:
            raise ValueError('Please set the correct dataset.')

    def map_labels(self, order_list, Y_set):
        """The function to map the labels according to the class order list.
        Args:
          order_list: the class order list.
          Y_set: the target labels before mapping
        Return:
          map_Y: the mapped target labels
        """
        map_Y = []
        for idx in Y_set:
            map_Y.append(order_list.index(idx))
        map_Y = np.array(map_Y)
        return map_Y

    def set_dataset(self):
        """The function to set the datasets.
        Returns:
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
          X_valid_total: an array that contains all validation samples
          Y_valid_total: an array that contains all validation labels 
        """
        if self.args.dataset == 'cifar100':
            X_train_total = np.array(self.trainset.data)
            Y_train_total = np.array(self.trainset.targets)
            X_valid_total = np.array(self.testset.data)
            Y_valid_total = np.array(self.testset.targets)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            X_train_total, Y_train_total = split_images_labels(self.trainset.imgs)
            X_valid_total, Y_valid_total = split_images_labels(self.testset.imgs)
        else:
            raise ValueError('Please set the correct dataset.')

        return X_train_total, Y_train_total, X_valid_total, Y_valid_total    

    def init_fusion_vars(self):
        """The function to initialize the aggregation weights."""
        self.fusion_vars = nn.ParameterList()
        if self.args.dataset == 'cifar100':
            # CIFAR-100, the number of blocks: 3
            if self.args.branch_mode == 'dual':
                # Dual branch mode, intialize the aggregation weights to 0.5
                for idx in range(3):
                    self.fusion_vars.append(nn.Parameter(torch.FloatTensor([0.5])))
            elif self.args.branch_mode == 'single':
                # Single branch mode, intialize the aggregation weights for the 1st branch to 1.0, and never update them
                for idx in range(3):
                    self.fusion_vars.append(nn.Parameter(torch.FloatTensor([1.0])))
            else:
                raise ValueError('Please set correct mode.')
            # Send the aggregation weights to GPU 
            self.fusion_vars.to(self.device)

        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # ImageNet, the number of blocks: 4
            if self.args.branch_mode == 'dual':
                # Dual branch mode, intialize the aggregation weights to 0.5
                for idx in range(4):
                    self.fusion_vars.append(nn.Parameter(torch.FloatTensor([0.5])))
            elif self.args.branch_mode == 'single':
                # Single branch mode, intialize the aggregation weights for the 1st branch to 1.0, and never update them
                for idx in range(4):
                    self.fusion_vars.append(nn.Parameter(torch.FloatTensor([1.0])))
            else:
                raise ValueError('Please set correct mode.')
            # Send the aggregation weights to GPU 
            self.fusion_vars.to(self.device)
        else:
            raise ValueError('Please set correct dataset.')

    def init_class_order(self):
        """The function to initialize the class order.
        Returns:
          order: an array for the class order
          order_list: a list for the class order
        """
        # Set the random seed according to the config
        np.random.seed(self.args.random_seed)
        # Set the name for the class order file
        order_name = osp.join(self.save_path, "seed_{}_{}_order.pkl".format(self.args.random_seed, self.args.dataset))
        # Print the name for the class order file
        print("Order name:{}".format(order_name))
        
        if osp.exists(order_name):
            # If we have already generated the class order file, load it
            print("Loading the saved class order")
            order = utils.misc.unpickle(order_name)
        else:
            # If we don't have the class order file, generate a new one
            print("Generating a new class order")
            order = np.arange(self.args.num_classes)
            np.random.shuffle(order)
            utils.misc.savepickle(order, order_name)
        # Transfer the array to a list
        order_list = list(order)
        # Print the class order
        print(order_list)
        return order, order_list

    def init_prototypes(self, dictionary_size, order, X_train_total, Y_train_total):
        """The function to intialize the prototypes.
           Please note that the prototypes here contains all training samples.
           alpha_dr_herding contains the indexes for the selected exemplars
        Args:
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
          order: the class order
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
        Returns:
          alpha_dr_herding: an empty array to store the indexes for the exemplars
          prototypes: an array contains all training samples for all phases
        """
        # Set an empty to store the indexes for the selected exemplars
        alpha_dr_herding  = np.zeros((int(self.args.num_classes/self.args.nb_cl), dictionary_size, self.args.nb_cl), np.float32)
        if self.args.dataset == 'cifar100':
            # CIFAR-100, directly load the tensors for the training samples
            prototypes = np.zeros((self.args.num_classes, dictionary_size, X_train_total.shape[1], X_train_total.shape[2], X_train_total.shape[3]))
            for orde in range(self.args.num_classes):
                prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # ImageNet, save the paths for the training samples if an array
            prototypes = [[] for i in range(self.args.num_classes)]
            for orde in range(self.args.num_classes):
                prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
            prototypes = np.array(prototypes)
        else:
            raise ValueError('Please set correct dataset.')
        return alpha_dr_herding, prototypes

    def init_current_phase_model(self, iteration, start_iter, b1_model, b2_model):
        """The function to intialize the models for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          b1_model: the 1st branch model from last phase
          b2_model: the 2nd branch model from last phase 
        Returns:
          b1_model: the 1st branch model from the current phase
          b2_model: the 2nd branch model from the current phase
          ref_model: the 1st branch model from last phase (frozen, not trainable)
          ref_b2_model: the 2nd branch model from last phase (frozen, not trainable)
          the_lambda_mult, cur_the_lambda: the_lambda-related parameters for the current phase
          last_iter: the iteration index for last phase
        """
        if iteration == start_iter:
            # The 0th phase
            # Set the index for last phase to 0
            last_iter = 0
            # For the 0th phase, use the conventional ResNet
            b1_model = self.network(num_classes=self.args.nb_cl_fg)
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features = b1_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            # The 2nd branch and the reference model are not used, set them to None
            ref_model = None
            b2_model = None
            ref_b2_model = None
            the_lambda_mult = None
        elif iteration == start_iter+1:
            # The 1st phase
            # Update the index for last phase
            last_iter = iteration
            # Copy and freeze the 1st branch model
            ref_model = copy.deepcopy(b1_model)
            # Copy and freeze the aggregation weights
            self.ref_fusion_vars = copy.deepcopy(self.fusion_vars)
            # Set the 1st branch for the 1st phase
            if self.args.branch_1 == 'ss':
                b1_model = self.network_mtl(num_classes=self.args.nb_cl_fg)
            else:
                b1_model = self.network(num_classes=self.args.nb_cl_fg)
            # Load the model parameters trained last phase to the current phase model
            ref_dict = ref_model.state_dict()
            tg_dict = b1_model.state_dict()
            tg_dict.update(ref_dict)
            b1_model.load_state_dict(tg_dict)
            b1_model.to(self.device)
            # Set the 2nd branch for the 1st phase
            if self.args.branch_2 == 'ss':
                b2_model = self.network_mtl(num_classes=self.args.nb_cl_fg)
            else:
                b2_model = self.network(num_classes=self.args.nb_cl_fg)
            # Load the model parameters trained last phase to the current phase model
            b2_dict = b2_model.state_dict()
            b2_dict.update(ref_dict)
            b2_model.load_state_dict(b2_dict)
            b2_model.to(self.device)
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features = b1_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
            # Set the final FC layer for classification
            new_fc.fc1.weight.data = b1_model.fc.weight.data
            new_fc.sigma.data = b1_model.fc.sigma.data
            b1_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = out_features*1.0 / self.args.nb_cl
            # The 2nd branch doesn't have reference model, set it to None
            ref_b2_model = None
        else:
            # The i-th phase, i>=2
            # Update the index for last phase
            last_iter = iteration
            # Copy and freeze the 1st branch model
            ref_model = copy.deepcopy(b1_model)
            # Copy and freeze the aggregation weights
            self.ref_fusion_vars = copy.deepcopy(self.fusion_vars)
            # Copy and freeze the 2nd branch model
            ref_b2_model = copy.deepcopy(b2_model)
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features1 = b1_model.fc.fc1.out_features
            out_features2 = b1_model.fc.fc2.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features1+out_features2)
            # Set the final FC layer for classification
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = b1_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = b1_model.fc.fc2.weight.data
            new_fc.sigma.data = b1_model.fc.sigma.data
            b1_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = (out_features1+out_features2)*1.0 / (self.args.nb_cl)

        # Update the current lambda value for the current phase
        if iteration > start_iter:
            cur_the_lambda = self.args.the_lambda * math.sqrt(the_lambda_mult)
        else:
            cur_the_lambda = self.args.the_lambda
        return b1_model, b2_model, ref_model, ref_b2_model, the_lambda_mult, cur_the_lambda, last_iter

    def init_current_phase_dataset(self, iteration, start_iter, last_iter, order, order_list, \
        X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
        X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
        X_protoset_cumuls, Y_protoset_cumuls):
        """The function to intialize the dataset for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          last_iter: the iteration index for last phase
          order: the array for the class order
          order_list: the list for the class order
          X_train_total: the array that contains all training samples
          Y_train_total: the array that contains all training labels 
          X_valid_total: then array that contains all validation samples
          Y_valid_total: the array that contains all validation labels 
          X_train_cumuls: the array that contains old training samples
          Y_train_cumuls: the array that contains old training labels 
          X_valid_cumuls: the array that contains old validation samples
          Y_valid_cumuls: the array that contains old validation labels 
          X_protoset_cumuls: the array that contains old exemplar samples
          Y_protoset_cumuls: the array that contains old exemplar labels
        Returns:
          indices_train_10: the indexes of new-class samples
          X_train_cumuls: an array that contains old training samples, updated
          Y_train_cumuls: an array that contains old training labels, updated 
          X_valid_cumuls: an array that contains old validation samples, updated
          Y_valid_cumuls: an array that contains old validation labels, updated
          X_protoset_cumuls: an array that contains old exemplar samples, updated
          Y_protoset_cumuls: an array that contains old exemplar labels, updated
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
          X_valid_ori: an array that contains the 0th-phase validation samples, updated
          Y_valid_ori: an array that contains the 0th-phase validation labels, updated
          X_protoset: an array that contains the exemplar samples
          Y_protoset: an array that contains the exemplar labels
        """

        # Get the indexes of new-class samples (including training and test)
        indices_train_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_train_total])
        indices_test_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_valid_total])
                
        # Get the samples according to the indexes
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]

        # Add the new-class samples to the cumulative X array
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)
        X_train_cumul = np.concatenate(X_train_cumuls)

        # Get the labels according to the indexes, and add them to the cumulative Y array
        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        Y_train_cumul = np.concatenate(Y_train_cumuls)

        if iteration == start_iter:
            # Save the 0th-phase validation samples and labels 
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            # Update the exemplar set
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            # Create the training samples/labels for the current phase training
            X_train = np.concatenate((X_train,X_protoset),axis=0)
            Y_train = np.concatenate((Y_train,Y_protoset))

        # Generate the mapped labels, according the order list
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
    
        # Return different variables for different phases
        if iteration == start_iter:
            return indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, X_train_cumuls, Y_valid_cumuls, \
                X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, X_train, map_Y_train, \
                map_Y_valid_cumul, X_valid_ori, Y_valid_ori
        else:
            return indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, X_train_cumuls, Y_valid_cumuls, \
                X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, X_train, map_Y_train, \
                map_Y_valid_cumul, X_protoset, Y_protoset

    def imprint_weights(self, b1_model, b2_model, iteration, is_start_iteration, X_train, map_Y_train, dictionary_size):
        """The function to imprint FC classifier's weights 
        Args:
          b1_model: the 1st branch model from last phase
          b2_model: the 2nd branch model from last phase 
          iteration: the iteration index 
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
        Returns:
          b1_model: the 1st branch model from the current phase, the FC classifier is updated
        """
        if self.args.dataset == 'cifar100':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = b1_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # tg_feature_model is b1_model without the FC layer
            tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = b1_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                # Set a temporary dataloader for the current class
                self.evalset.data = X_train[cls_indices].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the feature maps using the current model
                cls_features = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                    tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps 
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            b1_model.to(self.device)
            b1_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = b1_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # tg_feature_model is b1_model without the FC layer
            tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = b1_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])<=dictionary_size)
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=2)
                num_samples = len(X_train[cls_indices])
                # Compute the feature maps using the current model
                cls_features = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                    tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            b1_model.to(self.device)
            b1_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        else:
            raise ValueError('Please set correct dataset.')
        return b1_model

    def update_train_and_valid_loader(self, X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul, \
        iteration, start_iter):
        """The function to update the dataloaders
        Args:
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          X_valid_cumuls: an array that contains old validation samples
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
          iteration: the iteration index 
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
        Returns:
          trainloader: the training dataloader
          testloader: the test dataloader
        """
        print('Setting the dataloaders ...')
        if self.args.dataset == 'cifar100':
            # Set the training dataloader
            self.trainset.data = X_train.astype('uint8')
            self.trainset.targets = map_Y_train
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                shuffle=True, num_workers=self.args.num_workers)
            # Set the test dataloader
            self.testset.data = X_valid_cumul.astype('uint8')
            self.testset.targets = map_Y_valid_cumul
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Set the training dataloader
            current_train_imgs = merge_images_labels(X_train, map_Y_train)
            self.trainset.imgs = self.trainset.samples = current_train_imgs
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
            # Set the test dataloader
            current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.testset.imgs = self.testset.samples = current_test_imgs
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set the correct dataset.')
        return trainloader, testloader

    def set_optimizer(self, iteration, start_iter, b1_model, ref_model, b2_model, ref_b2_model):
        """The function to set the optimizers for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          b1_model: the 1st branch model from the current phase
          b2_model: the 2nd branch model from the current phase 
          ref_model: the 1st branch model from last phase (frozen, not trainable)
          ref_b2_model: the 2nd branch model from last phase (frozen, not trainable)
        Returns:
          tg_optimizer: the optimizer for b1_model and b2_model
          tg_lr_scheduler: the learning rate decay scheduler for b1_model and b2_model
          fusion_optimizer: the optimizer for the aggregation weights
          fusion_lr_scheduler: the learning rate decay scheduler for the aggregation weights
        """
        if iteration > start_iter: 
            # The i-th phase (i>=2)
            
            # Transfer the forzen reference models to GPU
            if ref_model is not None:                  
                ref_model = ref_model.to(self.device)
            if ref_b2_model is not None: 
                ref_b2_model = ref_b2_model.to(self.device)

            # Freeze the FC weights for old classes, get the parameters for the 1st branch
            ignored_params = list(map(id, b1_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, b1_model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)

            # Get the parameters for the 2nd branch
            b2_params = b2_model.parameters()

            if self.args.branch_1 == 'fixed':
                # If the 1st branch is fixed, set the learning rate to zero
                branch1_lr = 0.0
                branch1_weight_decay = 0
            else:
                # If the 1st branch is not fixed, using the learning rate in the config
                branch1_lr = self.args.base_lr2
                branch1_weight_decay = self.args.custom_weight_decay 

            if self.args.branch_2 == 'fixed':
                # If the 2nd branch is fixed, set the learning rate to zero
                branch2_lr = 0.0
                branch2_weight_decay = 0
            else:
                # If the 2nd branch is not fixed, using the learning rate in the config
                branch2_lr = self.args.base_lr2
                branch2_weight_decay = self.args.custom_weight_decay             

            # Combine the parameters and the learning rates
            tg_params_new =[{'params': base_params, 'lr': branch1_lr, 'weight_decay': branch1_weight_decay}, \
                {'params': b2_params, 'lr': branch2_lr, 'weight_decay': branch2_weight_decay}, \
                {'params': b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

            # Transfer the 1st branch model to the GPU
            b1_model = b1_model.to(self.device)
            
            # Set the optimizer for b1_model and b2_model
            tg_optimizer = optim.SGD(tg_params_new, lr=self.args.base_lr2, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
         
            # Set the optimizer for the aggregation weights
            if self.args.branch_mode == 'dual':
                # Dual branch mode, load the learning rate for the aggregation weights from the config
                fusion_optimizer = optim.SGD(self.fusion_vars, lr=self.args.fusion_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            elif self.args.branch_mode == 'single':
                # Single branch mode, freeze the aggregation weight by setting the learning rates to zeros
                fusion_optimizer = optim.SGD(self.fusion_vars, lr=0.0, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            else:
                raise ValueError('Please set the correct mode.')
        else:
            # The 0th phase
            # For the 0th phase, we train conventional CNNs, so we don't need to update the aggregation weights
            tg_params = b1_model.parameters()
            b1_model = b1_model.to(self.device)
            tg_optimizer = optim.SGD(tg_params, lr=self.args.base_lr1, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            if self.args.branch_mode == 'dual':
                fusion_optimizer = optim.SGD(self.fusion_vars, lr=self.args.fusion_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            elif self.args.branch_mode == 'single':
                fusion_optimizer = optim.SGD(self.fusion_vars, lr=0.0, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            else:
                raise ValueError('Please set the correct mode.')

        # Set the learning rate decay scheduler
        if self.args.dataset == 'cifar100':
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)
            fusion_lr_scheduler = lr_scheduler.MultiStepLR(fusion_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            tg_lr_scheduler = lr_scheduler.CosineAnnealingLR(tg_optimizer, self.args.epochs)        
            fusion_lr_scheduler = lr_scheduler.MultiStepLR(fusion_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)
        else:
            raise ValueError('Please set the correct dataset.')    

        return tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler

    def gen_balanced_loader(self, X_train_total, Y_train_total, indices_train_10, X_protoset, Y_protoset, order_list):
        """The function to generate the balanced loader
        Args:
          X_train_total: the array that contains all training samples
          Y_train_total: the array that contains all training labels 
          indices_train_10: the indexes of new-class samples
          X_protoset: an array that contains the exemplar samples
          Y_protoset: an array that contains the exemplar labels
        Return:
          balancedloader: the balanced dataloader for the exemplars
        """
        if self.args.dataset == 'cifar100':
            # Load the training samples for the current phase
            X_train_this_step = X_train_total[indices_train_10]
            Y_train_this_step = Y_train_total[indices_train_10]

            # Using random index to select the exemplars for the current phase (before training)
            the_idx = np.random.randint(0,len(X_train_this_step),size=self.args.nb_cl*self.args.nb_protos)
            
            # Merge the current-phase exemplars and the old exemplars
            X_balanced_this_step = np.concatenate((X_train_this_step[the_idx],X_protoset),axis=0)
            Y_balanced_this_step = np.concatenate((Y_train_this_step[the_idx],Y_protoset),axis=0)
            map_Y_train_this_step = np.array([order_list.index(i) for i in Y_balanced_this_step])
            # Build the balanced dataloader
            self.balancedset.data = X_balanced_this_step.astype('uint8')
            self.balancedset.targets = map_Y_train_this_step               
            balancedloader = torch.utils.data.DataLoader(self.balancedset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':  
            # Load the training samples for the current phase
            X_train_this_step = X_train_total[indices_train_10]
            Y_train_this_step = Y_train_total[indices_train_10]

            # Using random index to select the exemplars for the current phase (before training)
            the_idx = np.random.randint(0,len(X_train_this_step),size=self.args.nb_cl*self.args.nb_protos)
            X_balanced_this_step = np.concatenate((X_train_this_step[the_idx],X_protoset),axis=0)
            Y_balanced_this_step = np.concatenate((Y_train_this_step[the_idx],Y_protoset),axis=0)

            # Merge the current-phase exemplars and the old exemplars
            map_Y_train_this_step = np.array([order_list.index(i) for i in Y_balanced_this_step])

            # Build the balanced dataloader
            current_train_imgs = merge_images_labels(X_balanced_this_step, map_Y_train_this_step)
            self.balancedset.imgs = self.balancedset.samples = current_train_imgs
            balancedloader = torch.utils.data.DataLoader(self.balancedset, batch_size=self.args.test_batch_size, \
            shuffle=False, num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set the correct dataset.')
        return balancedloader

    def compute_acc(self, class_means, order, order_list, b1_model, b2_model, X_protoset_cumuls, Y_protoset_cumuls, \
        X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul):
        """The function to compute the accuracy
        Args:
          class_means: the mean values for each class
          order: the array for the class order
          order_list: the list for the class order
          b1_model: the 1st branch model from the current phase
          b2_model: the 2nd branch model from the current phase 
          X_protoset_cumuls: the array that contains old exemplar samples
          Y_protoset_cumuls: the array that contains old exemplar labels
          X_valid_ori: the array that contains the 0th-phase validation samples, updated
          Y_valid_ori: the array that contains the 0th-phase validation labels, updated
          X_valid_cumuls: the array that contains old validation samples
          Y_valid_cumuls: the array that contains old validation labels 
          iteration: the iteration index
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          top1_acc_list_ori: the list to store the results for the 0th classes
          top1_acc_list_cumul: the list to store the results for the current phase
        Returns:
          top1_acc_list_ori: the list to store the results for the 0th classes, updated
          top1_acc_list_cumul: the list to store the results for the current phase, updated
        """

        # Get tg_feature_model, which is a model copied from b1_model, without the FC layer
        tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
        # Get the class mean values for all seen classes
        current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]

        # Get mapped labels for the 0-th phase data, according the the order list
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the 0-th phase classes...')
        # Set a temporary dataloader for the 0-th phase data
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_ori.astype('uint8')
            self.evalset.targets = map_Y_valid_ori
            pin_memory = False
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
            current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
            self.evalset.imgs = self.evalset.samples = current_eval_set
            pin_memory = True
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)
        # Compute the accuracies for the 0-th phase test data
        ori_acc, fast_fc = compute_accuracy(self.args, self.fusion_vars, b1_model, b2_model, tg_feature_model, \
            current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, \
            order_list, is_start_iteration=is_start_iteration)
        # Add the results to the array, which stores all previous results
        top1_acc_list_ori[iteration, :, 0] = np.array(ori_acc).T
        # Write the results to tensorboard
        self.train_writer.add_scalar('ori_acc/fc', float(ori_acc[0]), iteration)
        self.train_writer.add_scalar('ori_acc/proto', float(ori_acc[1]), iteration)
        # Get mapped labels for the current-phase data, according the the order list
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        # Set a temporary dataloader for the current-phase data
        print('Computing cumulative accuracy...')
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_cumul.astype('uint8')
            self.evalset.targets = map_Y_valid_cumul
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':  
            current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.evalset.imgs = self.evalset.samples = current_eval_set
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)    
        # Compute the accuracies for the current-phase data    
        cumul_acc, _ = compute_accuracy(self.args, self.fusion_vars, b1_model, b2_model, tg_feature_model, \
            current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, \
            is_start_iteration=is_start_iteration, fast_fc=fast_fc)
        # Add the results to the array, which stores all previous results
        top1_acc_list_cumul[iteration, :, 0] = np.array(cumul_acc).T
        # Write the results to tensorboard
        self.train_writer.add_scalar('cumul_acc/fc', float(cumul_acc[0]), iteration)
        self.train_writer.add_scalar('cumul_acc/proto', float(cumul_acc[1]), iteration)

        return top1_acc_list_ori, top1_acc_list_cumul

    def set_exemplar_set(self, b1_model, b2_model, is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes):
        """The function to select the exemplars
        Args:
          b1_model: the 1st branch model from the current phase
          b2_model: the 2nd branch model from the current phase 
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          iteration: the iteration index
          last_iter: the iteration index for last phase
          order: the array for the class order
          alpha_dr_herding: the empty array to store the indexes for the exemplars
          prototypes: the array contains all training samples for all phases
        Returns:
          X_protoset_cumuls: an array that contains old exemplar samples
          Y_protoset_cumuls: an array that contains old exemplar labels
          class_means: the mean values for each class
          alpha_dr_herding: the empty array to store the indexes for the exemplars, updated
        """
        # Use the dictionary size defined in this class-incremental learning class
        dictionary_size = self.dictionary_size
        if self.args.dynamic_budget:
            # Using dynamic exemplar budget, i.e., 20 exemplars each class. In this setting, the total memory budget is increasing
            nb_protos_cl = self.args.nb_protos
        else:
            # Using fixed exemplar budget. The total memory size is unchanged
            nb_protos_cl = int(np.ceil(self.args.nb_protos*100./self.args.nb_cl/(iteration+1)))
        # Get tg_feature_model, which is a model copied from b1_model, without the FC layer
        tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
        # Get the shape for the feature maps
        num_features = b1_model.fc.in_features
        if self.args.dataset == 'cifar100':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                self.evalset.data = prototypes[iter_dico].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the features for the current class          
                mapped_prototypes = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                    tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                num_samples = len(prototypes[iter_dico])            
                # Compute the features for the current class  
                mapped_prototypes = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                    tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)

                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
        else:
            raise ValueError('Please set the correct dataset.')
        # Set two empty lists for the exemplars and the labels 
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        if self.args.dataset == 'cifar100':
            class_means = np.zeros((64,100,2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.data.shape[0]
                    mapped_prototypes = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                        tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                        tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    # Using the indexes selected by herding
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico,np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    # Compute the class mean values                  
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':            
            class_means = np.zeros((num_features, self.args.num_classes, 2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    current_eval_set = merge_images_labels(prototypes[iteration2*self.args.nb_cl+iter_dico], \
                        np.zeros(len(prototypes[iteration2*self.args.nb_cl+iter_dico])))
                    self.evalset.imgs = self.evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                    num_samples = len(prototypes[iteration2*self.args.nb_cl+iter_dico])
                    mapped_prototypes = compute_features(self.args, self.fusion_vars, b1_model, b2_model, \
                        tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    D2 = D
                    # Using the indexes selected by herding
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    assert((alph[num_samples:]==0).all())
                    alph = alph[:num_samples]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico][np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    # Compute the class mean values   
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(num_samples)/num_samples
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        else:
            raise ValueError('Please set the correct dataset.')

        # Save the class mean values   
        torch.save(class_means, osp.join(self.save_path, 'iter_{}_class_means.pth'.format(iteration)))
        return X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding
