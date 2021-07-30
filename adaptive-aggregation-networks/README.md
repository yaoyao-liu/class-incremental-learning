## Adaptive Aggregation Networks for Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

\[[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Adaptive_Aggregation_Networks_for_Class-Incremental_Learning_CVPR_2021_paper.pdf)\] \[[Project Page](https://class-il.mpi-inf.mpg.de/)\] \[[GitLab@MPI](https://gitlab.mpi-klsb.mpg.de/yaoyaoliu/adaptive-aggregation-networks)\] 

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Running Experiments](#running-experiments)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Introduction

Class-Incremental Learning (CIL) aims to learn a classification model with the number of classes increasing phase-by-phase. The inherent problem in CIL is the stability-plasticity dilemma between the learning of old and new classes, i.e., high-plasticity models easily forget old classes but high-stability models are weak to learn new classes. We alleviate this issue by proposing a novel network architecture called Adaptive Aggregation Networks (AANets) in which we explicitly build two residual blocks at each residual level (taking ResNet as the baseline architecture): a stable block and a plastic block. We aggregate the output feature maps from these two blocks and then feed the results to the next-level blocks. We meta-learn the aggregating weights in order to dynamically optimize and balance between two types of blocks, i.e., between stability and plasticity. We conduct extensive experiments on three CIL benchmarks: CIFAR-100, ImageNet-Subset, and ImageNet, and show that many existing CIL methods can be straightforwardly incorporated on the architecture of AANets to boost their performance. 

<p align="center">
    <img src="https://images.yyliu.net/AANets-1.png" width="800"/>
</p>

> Figure: Conceptual illustrations of different CIL methods. (a) Conventional methods use all available data (imbalanced classes) to train the model (Rebuffi et al., 2017; Hou et al., 2019) (b) Castro et al. (2018), Hou et al. (2019) and Douillard et al. (2020) follow the convention but add a fine-tuning step using the balanced set of exemplars. (c) Our AANets approach uses all available data to update the plastic and stable blocks, and use the balanced set of exemplars to meta-learn the aggregating weights. We continuously update these weights such as to dynamically balance between plastic and stable blocks, i.e., between plasticity and stability

### Getting Started

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name AANets-PyTorch python=3.6
conda activate AANets-PyTorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install tqdm scipy sklearn tensorboardX Pillow==6.2.2
```

### Running Experiments
#### Running Experiments w/ AANets on CIFAR-100

[LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning) w/ AANets
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100
```

[iCaRL](https://github.com/hshustc/CVPR19_Incremental_Learning) w/ AANets
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 
```

#### Running Baseline Experiments on CIFAR-100

[LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning) w/o AANets, dual branch
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=free --branch_2=free ---fusion_lr=0.0 -dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100
```

[iCaRL](https://github.com/hshustc/CVPR19_Incremental_Learning) w/o AANets, dual branch
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100 
```

[LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning) w/o AANets, single branch
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free -dataset=cifar100
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=single --branch_1=free --dataset=cifar100
```

[iCaRL](https://github.com/hshustc/CVPR19_Incremental_Learning) w/o AANets, single branch
```bash
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=cifar100 
python main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --baseline=icarl --branch_mode=single --branch_1=free --dataset=cifar100 
```

#### Running Experiments on ImageNet
To run the experiments on ImageNet (including ImageNet-Subset and ImageNet-Full), you need to change the hyperparameters according to [this file](https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/imagenet-class-incremental/cbf_class_incremental_cosine_imagenet.py). 

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{Liu2020AANets,
  author    = {Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},
  title     = {Adaptive Aggregation Networks for Class-Incremental Learning},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2544-2553},
  year      = {2021}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)
