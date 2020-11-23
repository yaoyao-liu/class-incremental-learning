## Meta-Aggregating Networks for Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square)](https://pytorch.org/)

\[[arXiv preprint](https://arxiv.org/pdf/2010.05063.pdf)\]

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Running Experiments](#running-experiments)

### Introduction

Class-Incremental Learning (CIL) aims to learn a classification model with the number of classes increasing phase-by-phase. The inherent problem in CIL is the stability-plasticity dilemma between the learning of old and new classes, i.e., high-plasticity models easily forget old classes but high-stability models are weak to learn new classes. We alleviate this issue by proposing a novel network architecture called Meta-Aggregating Networks (MANets) in which we explicitly build two residual blocks at each residual level (taking ResNet as the baseline architecture): a stable block and a plastic block. We aggregate the output feature maps from these two blocks and then feed the results to the next-level blocks. We meta-learn the aggregating weights in order to dynamically optimize and balance between two types of blocks, i.e., between stability and plasticity. We conduct extensive experiments on three CIL benchmarks: CIFAR-100, ImageNet-Subset, and ImageNet, and show that many existing CIL methods can be straightforwardly incorporated on the architecture of MANets to boost their performance. 

<p align="center">
    <img src="https://yyliu.net/images/misc/MANets.png" width="800"/>
</p>

> Figure: Conceptual illustrations of different CIL methods. (a) Conventional methods use all available data (imbalanced classes) to train the model (Rebuffi et al., 2017; Hou et al., 2019) (b) Castro et al. (2018), Hou et al. (2019) and Douillard et al. (2020) follow the convention but add a fine-tuning step using the balanced set of exemplars. (c) Our MANets approach uses all available data to update the plastic and stable blocks, and use the balanced set of exemplars to meta-learn the aggregating weights. We continuously update these weights such as to dynamically balance between plastic and stable blocks, i.e., between plasticity and stability

### Getting Started

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name MANets-PyTorch python=3.6
conda activate MANets-PyTorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install tqdm tensorboardX Pillow==6.2.2
```

### Running Experiments

CIFAR-100 (basline: iCaRL, Rebuffi et al., 2017)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
```

CIFAR-100 (basline: LUCIR, Hou et al., 2019)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
```

ImageNet-Subset (basline: iCaRL, Rebuffi et al., 2017)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
```

ImageNet-Subset (basline: LUCIR, Hou et al., 2019)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 
```

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{Liu2020MANets
  author    = {Liu, Yaoyao and
               Schiele, Bernt and
               Sun, Qianru},
  title     = {Meta-Aggregating Networks for Class-Incremental Learning},
  journal   = {arXiv},
  volume    = {2010.05063},
  year      = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)
