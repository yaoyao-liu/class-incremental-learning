# Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)

### News

- We update the code for [Adaptive Aggregation Networks](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks) (accepted to CVPR 2021), which achieve SOTA performance on class-incremental learning tasks. Detailed comments are added for most of the functions and classes. 

### Papers

- Adaptive Aggregation Networks for Class-Incremental Learning,
CVPR 2021. \[[PDF](https://arxiv.org/pdf/2010.05063.pdf)\] \[[Project](https://class-il.mpi-inf.mpg.de/)\]

- Mnemonics Training: Multi-Class Incremental Learning without Forgetting,
CVPR 2020. \[[PDF](https://arxiv.org/pdf/2002.10211.pdf)\] \[[Project](https://class-il.mpi-inf.mpg.de/mnemonics/)\]

### Citations

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{Liu2020AANets,
  author    = {Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},
  title     = {Adaptive Aggregation Networks for Class-Incremental Learning},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```

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
