# Mnemonics Training

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

\[[PDF](https://arxiv.org/pdf/2002.10211.pdf)\] \[[Project Page](https://class-il.mpi-inf.mpg.de/mnemonics-training/)\]

## Requirements

See the versions for the requirements [here](https://yyliu.net/files/mnemonics_packages.txt).

## Download the Datasest

See the details [here](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks#download-the-datasets).


## Running Experiments

### Running experiments for baselines

```bash
cd ./mnemonics-training/1_train
python main.py --method=baseline --nb_cl=10
python main.py --method=baseline --nb_cl=5
python main.py --method=baseline --nb_cl=2
```

### Running experiments for our method

```bash
cd ./mnemonics-training/1_train
python main.py --method=mnemonics --nb_cl=10
python main.py --method=mnemonics --nb_cl=5
python main.py --method=mnemonics --nb_cl=2
```

### Performance

#### Average accuracy (%)

| Method          | Dataset   | 5-phase     | 10-phase     | 25-phase    | 
| ----------      | --------- | ----------  | ----------   |------------ |
| [LwF](https://arxiv.org/abs/1606.09282)  | CIFAR-100 | 52.44  | 48.47   | 45.75 |
| [LwF](https://arxiv.org/abs/1606.09282) w/ ours  | CIFAR-100 | 54.21  | 52.72   | 51.59 |
| [iCaRL](https://arxiv.org/abs/1611.07725)  | CIFAR-100 | 58.03  | 53.01  | 48.47 |
| [iCaRL](https://arxiv.org/abs/1611.07725) w/ ours | CIFAR-100 | 60.01  | 57.37   | 54.13 |

#### Forgetting rate (%, lower is better)

| Method          | Dataset   | 5-phase     | 10-phase     | 25-phase    | 
| ----------      | --------- | ----------  | ----------   |------------ |
| [LwF](https://arxiv.org/abs/1606.09282)  | CIFAR-100 | 45.02  | 42.50   | 39.86 |
| [LwF](https://arxiv.org/abs/1606.09282) w/ ours  | CIFAR-100 | 40.00  | 36.50   | 34.25 |
| [iCaRL](https://arxiv.org/abs/1611.07725)  | CIFAR-100 | 32.87  | 32.98 | 36.32 |
| [iCaRL](https://arxiv.org/abs/1611.07725) w/ ours | CIFAR-100 | 25.93  | 26.92   | 28.92 |

## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{liu2020mnemonics,
author    = {Liu, Yaoyao and Su, Yuting and Liu, An{-}An and Schiele, Bernt and Sun, Qianru},
title     = {Mnemonics Training: Multi-Class Incremental Learning without Forgetting},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
pages     = {12245--12254},
year      = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)

* [Dataset Distillation](https://github.com/SsnL/dataset-distillation)

* [Generative Teaching Networks](https://github.com/uber-research/GTN)
