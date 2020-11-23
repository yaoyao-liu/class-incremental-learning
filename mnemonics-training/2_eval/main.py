#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import numpy as np
from trainer.train import Trainer
from utils.gpu_tools import occupy_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0') # GPU id 
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--data_dir', default='data/seed_1993_subset_100_imagenet/data', type=str)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in first group')
    parser.add_argument('--nb_cl', default=10, type=int, help='Classes per group')
    parser.add_argument('--nb_protos', default=20, type=int, help='Number of prototypes per class at the end')
    parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
    parser.add_argument('--epochs', default=160, type=int, help='Epochs')
    parser.add_argument('--T', default=2, type=float, help='Temporature for distialltion')
    parser.add_argument('--beta', default=0.25, type=float, help='Beta for distialltion')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_fg', action='store_true', help='resume first group from checkpoint')
    parser.add_argument('--ckpt_dir_fg', type=str, default='-')
    parser.add_argument('--dynamic_budget', action='store_true', help='fix budget')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--ckpt_label', type=str, default='exp01')
    parser.add_argument('--use_mtl', action='store_true', help='using mtl weights')
    parser.add_argument('--num_workers', default=2, type=int, help='the number of workers for loading data')
    parser.add_argument('--load_iter', default=0, type=int)
    parser.add_argument('--mimic_score', action='store_true', help='To mimic scores for cosine embedding')
    parser.add_argument('--lw_ms', default=1, type=float, help='loss weight for mimicking score')
    parser.add_argument('--rs_ratio', default=0, type=float, help='The ratio for resample')
    parser.add_argument('--imprint_weights', action='store_true', help='Imprint the weights for novel classes')
    parser.add_argument('--less_forget', action='store_true', help='Less forgetful')
    parser.add_argument('--lamda', default=5, type=float, help='Lamda for LF')
    parser.add_argument('--adapt_lamda', action='store_true', help='Adaptively change lamda')
    parser.add_argument('--dist', default=0.5, type=float, help='Dist for MarginRankingLoss')
    parser.add_argument('--K', default=2, type=int, help='K for MarginRankingLoss')
    parser.add_argument('--lw_mr', default=1, type=float, help='loss weight for margin ranking loss')
    parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--base_lr1', default=0.1, type=float)
    parser.add_argument('--base_lr2', default=0.1, type=float)
    parser.add_argument('--lr_factor', default=0.1, type=float)
    parser.add_argument('--custom_weight_decay', default=5e-4, type=float)
    parser.add_argument('--custom_momentum', default=0.9, type=float)
    parser.add_argument('--load_ckpt_prefix', type=str, default='-')
    parser.add_argument('--load_order', type=str, default='-')
    parser.add_argument('--add_str', default=None, type=str)

    the_args = parser.parse_args()
    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    print(the_args)

    np.random.seed(the_args.random_seed)

    if not os.path.exists('./logs/cifar100_nfg50_ncls2_nproto20_mtl_exp01'):
        print('Download checkpoints from Google Drive.')
        os.system('sh ./script/download_ckpt.sh')

    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    occupy_memory(the_args.gpu)
    print('Occupy GPU memory in advance.')

    trainer = Trainer(the_args)
    trainer.eval()





