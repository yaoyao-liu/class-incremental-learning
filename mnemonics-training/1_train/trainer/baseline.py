import copy
import os
import torch
import torchvision
import warnings
import math
import utils.misc
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_linear as modified_linear
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from utils.compute_features import compute_features
from utils.process_mnemonics import process_mnemonics
from utils.compute_accuracy import compute_accuracy
from trainer.incremental import incremental_train_and_eval
from utils.misc import *
from utils.process_fp import process_inputs_fp
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self, the_args):
        self.args = the_args
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + self.args.dataset + '_nfg' + str(self.args.nb_cl_fg) + '_ncls' + str(self.args.nb_cl) + '_nproto' + str(self.args.nb_protos) 
        self.save_path += '_' + self.args.method        
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])
        self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])
        self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
        self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
        self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_test)
        self.network = modified_resnet_cifar.resnet32
        self.network_mtl = modified_resnetmtl_cifar.resnetmtl32
        self.lr_strat_first_phase = [int(160*0.5), int(160*0.75)]
        self.lr_strat = [int(self.args.epochs*0.5), int(self.args.epochs*0.75)]
        self.dictionary_size = self.args.dictionary_size

    def map_labels(self, order_list, Y_set):
        map_Y = []
        for idx in Y_set:
            map_Y.append(order_list.index(idx))
        map_Y = np.array(map_Y)
        return map_Y

    def train(self):
        self.train_writer = SummaryWriter(logdir=self.save_path)
        dictionary_size = self.dictionary_size
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 4, self.args.nb_runs))
        X_train_total = np.array(self.trainset.train_data)
        Y_train_total = np.array(self.trainset.train_labels)
        X_valid_total = np.array(self.testset.test_data)
        Y_valid_total = np.array(self.testset.test_labels)
        np.random.seed(1993)
        for iteration_total in range(self.args.nb_runs):
            order_name = osp.join(self.save_path, "seed_{}_{}_order_run_{}.pkl".format(1993, self.args.dataset, iteration_total))
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
        np.random.seed(self.args.random_seed)
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []
        alpha_dr_herding  = np.zeros((int(self.args.num_classes/self.args.nb_cl),dictionary_size,self.args.nb_cl),np.float32)
        prototypes = np.zeros((self.args.num_classes,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
        for orde in range(self.args.num_classes):
            prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1
        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            if iteration == start_iter:
                last_iter = 0
                tg_model = self.network(num_classes=self.args.nb_cl_fg)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("Out_features:", out_features)
                ref_model = None
                free_model = None
                ref_free_model = None
            elif iteration == start_iter+1:
                last_iter = iteration
                ref_model = copy.deepcopy(tg_model)
                print("Fusion Mode: "+self.args.fusion_mode)
                tg_model = self.network(num_classes=self.args.nb_cl_fg)
                ref_dict = ref_model.state_dict()
                tg_dict = tg_model.state_dict()
                tg_dict.update(ref_dict)
                tg_model.load_state_dict(tg_dict)
                tg_model.to(self.device)
                in_features = tg_model.fc.in_features
                out_features = tg_model.fc.out_features
                print("Out_features:", out_features)
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
                print("Out_features:", out_features1+out_features2)
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
                new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
                new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = (out_features1+out_features2)*1.0 / (self.args.nb_cl)
            if iteration > start_iter:
                cur_lamda = self.args.lamda * math.sqrt(lamda_mult)
            else:
                cur_lamda = self.args.lamda
            actual_cl = order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)]
            indices_train_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_train_total])
            indices_test_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_valid_total])
            X_train = X_train_total[indices_train_10]
            X_valid = X_valid_total[indices_test_10]
            X_valid_cumuls.append(X_valid)
            X_train_cumuls.append(X_train)
            X_valid_cumul = np.concatenate(X_valid_cumuls)
            X_train_cumul = np.concatenate(X_train_cumuls)
            Y_train = Y_train_total[indices_train_10]
            Y_valid = Y_valid_total[indices_test_10]
            Y_valid_cumuls.append(Y_valid)
            Y_train_cumuls.append(Y_train)
            Y_valid_cumul = np.concatenate(Y_valid_cumuls)
            Y_train_cumul = np.concatenate(Y_train_cumuls)
            if iteration == start_iter:
                X_valid_ori = X_valid
                Y_valid_ori = Y_valid
            else:
                X_protoset = np.concatenate(X_protoset_cumuls)
                Y_protoset = np.concatenate(Y_protoset_cumuls)
                if self.args.rs_ratio > 0:
                    scale_factor = (len(X_train) * self.args.rs_ratio) / (len(X_protoset) * (1 - self.args.rs_ratio))
                    rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
                    rs_num_samples = int(len(X_train) / (1 - self.args.rs_ratio))
                    print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
                X_train = np.concatenate((X_train,X_protoset),axis=0)
                Y_train = np.concatenate((Y_train,Y_protoset))
            print('Batch of classes number {0} arrives'.format(iteration+1))
            map_Y_train = np.array([order_list.index(i) for i in Y_train])
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
            is_start_iteration = (iteration == start_iter)
            if iteration > start_iter:
                old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
                average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
                tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
                num_features = tg_model.fc.in_features
                novel_embedding = torch.zeros((self.args.nb_cl, num_features))
                for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                    cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                    assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                    self.evalset.test_data = X_train[cls_indices].astype('uint8')
                    self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0])
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.test_data.shape[0]
                    cls_features = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                    cls_embedding = torch.mean(norm_features, dim=0)
                    novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
                tg_model.to(self.device)
                tg_model.fc.fc2.weight.data = novel_embedding.to(self.device)
            self.trainset.train_data = X_train.astype('uint8')
            self.trainset.train_labels = map_Y_train
            if iteration > start_iter and self.args.rs_ratio > 0 and scale_factor > 1:
                print("Weights from sampling:", rs_sample_weights)
                index1 = np.where(rs_sample_weights>1)[0]
                index2 = np.where(map_Y_train<iteration*self.args.nb_cl)[0]
                assert((index1==index2).all())
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
                trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size, shuffle=False, sampler=train_sampler, num_workers=self.args.num_workers)            
            else:
                trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                    shuffle=True, num_workers=self.args.num_workers)
            self.testset.test_data = X_valid_cumul.astype('uint8')
            self.testset.test_labels = map_Y_valid_cumul
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
            print('Max and min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
            print('Max and min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
            ckp_name = osp.join(self.save_path, 'run_{}_iteration_{}_model.pth'.format(iteration_total, iteration))
            ckp_name_free = osp.join(self.save_path, 'run_{}_iteration_{}_free_model.pth'.format(iteration_total, iteration))
            print('Checkpoint name:', ckp_name)
            if iteration==start_iter and self.args.resume_fg:
                print("Loading first group models from checkpoint")
                tg_model = torch.load(self.args.ckpt_dir_fg)
            elif self.args.resume and os.path.exists(ckp_name):
                print("Loading models from checkpoint")
                tg_model = torch.load(ckp_name)
            else:
                if iteration > start_iter:
                    ref_model = ref_model.to(self.device)
                    ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, tg_model.parameters())
                    base_params = filter(lambda p: p.requires_grad,base_params)
                    base_params = filter(lambda p: p.requires_grad,base_params)
                    tg_params_new =[{'params': base_params, 'lr': self.args.base_lr2, 'weight_decay': self.args.custom_weight_decay}, {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
                    tg_model = tg_model.to(self.device)
                    tg_optimizer = optim.SGD(tg_params_new, lr=self.args.base_lr2, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                else:
                    tg_params = tg_model.parameters()
                    tg_model = tg_model.to(self.device)
                    tg_optimizer = optim.SGD(tg_params, lr=self.args.base_lr1, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                if iteration > start_iter:
                    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)
                else:
                    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat_first_phase, gamma=self.args.lr_factor)           
                print("Incremental train")
                if iteration > start_iter:
                    tg_model = incremental_train_and_eval(self.args.epochs, tg_model, ref_model, free_model, ref_free_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, cur_lamda, self.args.dist, self.args.K, self.args.lw_mr)   
                else:                    
                    tg_model = incremental_train_and_eval(self.args.epochs, tg_model, ref_model, free_model, ref_free_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, cur_lamda, self.args.dist, self.args.K, self.args.lw_mr)            
                torch.save(tg_model, ckp_name)
            if self.args.dynamic_budget:
                nb_protos_cl = self.args.nb_protos
            else:
                nb_protos_cl = int(np.ceil(self.args.nb_protos*100./self.args.nb_cl/(iteration+1)))
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                self.evalset.test_data = prototypes[iter_dico].astype('uint8')
                self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.test_data.shape[0]            
                mapped_prototypes = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
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
            X_protoset_cumuls = []
            Y_protoset_cumuls = []
            class_means = np.zeros((64,100,2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0]) #zero labels
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.test_data.shape[0]
                    mapped_prototypes = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico,np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
            current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]
            class_means = np.zeros((64,100,2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.test_labels = np.zeros(self.evalset.test_data.shape[0]) #zero labels
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.test_data.shape[0]
                    mapped_prototypes = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    self.evalset.test_data = prototypes[iteration2*self.args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
            torch.save(class_means, osp.join(self.save_path, 'run_{}_iteration_{}_class_means.pth'.format(iteration_total, iteration)))
            current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]
            is_start_iteration = (iteration == start_iter)
            map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
            print('Computing accuracy for first-phase classes')
            self.evalset.test_data = X_valid_ori.astype('uint8')
            self.evalset.test_labels = map_Y_valid_ori
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)
            ori_acc, fast_fc = compute_accuracy(tg_model, free_model,  tg_feature_model, current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=is_start_iteration, maml_lr=self.args.maml_lr, maml_epoch=self.args.maml_epoch)
            top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
            self.train_writer.add_scalar('ori_acc/LwF', float(ori_acc[0]), iteration)
            self.train_writer.add_scalar('ori_acc/iCaRL', float(ori_acc[1]), iteration)
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
            print('Computing accuracy for all seen classes')
            self.evalset.test_data = X_valid_cumul.astype('uint8')
            self.evalset.test_labels = map_Y_valid_cumul
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)        
            cumul_acc, _ = compute_accuracy(tg_model, free_model, tg_feature_model, current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=is_start_iteration, fast_fc=fast_fc, maml_lr=self.args.maml_lr, maml_epoch=self.args.maml_epoch)
            top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
            self.train_writer.add_scalar('cumul_acc/LwF', float(cumul_acc[0]), iteration)
            self.train_writer.add_scalar('cumul_acc/iCaRL', float(cumul_acc[1]), iteration)
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'run_{}_top1_acc_list_ori.pth'.format(iteration_total)))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'run_{}_top1_acc_list_cumul.pth'.format(iteration_total)))
        self.train_writer.close
        
