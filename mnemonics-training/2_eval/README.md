## Evaluation on our models

### Clone this repository
```bash
cd ~
git clone git@github.com:yaoyao-liu/mnemonics-training.git
```

### Processing the datasets

Process ImageNet-Sub and ImageNet:
```bash
cd ~/mnemonics-training/eval/process_imagenet
python generate_imagenet_subset.py
python generate_imagenet.py
```

### Download models

Download the models for CIFAR-100, ImageNet-Sub and ImageNet:
```bash
cd ~/mnemonics-training/eval
sh ./script/download_ckpt.sh
```
You may also download the checkpoints on [Google Drive](https://drive.google.com/file/d/1sKO2BOssWgTFBNZbM50qDzgk6wqg4_l8/view).

### Running the evaluation

Run evaluation code with our models：
```bash
cd ~/mnemonics-training/eval
sh run_eval.sh
```
