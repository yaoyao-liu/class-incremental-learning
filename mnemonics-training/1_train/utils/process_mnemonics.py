import torch
import torch.optim as optim
import torchvision
import time
import os
import argparse
import numpy as np

def tensor2im(input_image, imtype=np.uint8):
    mean = [0.5071,  0.4866,  0.4409]
    std = [0.2009,  0.1984,  0.2023]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().detach().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): 
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def process_mnemonics(X_protoset_cumuls, Y_protoset_cumuls, mnemonics_raw, mnemonics_label, order_list, nb_cl_fg, nb_cl, iteration, start_iter):
    mnemonics = mnemonics_raw[0]
    mnemonics_array_new = np.zeros((len(mnemonics), len(mnemonics[0]), 32, 32, 3))
    mnemonics_list = []
    mnemonics_label_list = []
    for idx in range(len(mnemonics)):
        this_mnemonics = []
        for sub_idx in range(len(mnemonics[idx])):
            processed_img = tensor2im(mnemonics[idx][sub_idx])
            mnemonics_array_new[idx][sub_idx] = processed_img   
    diff = len(X_protoset_cumuls) - len(mnemonics_array_new)
    for idx in range(len(mnemonics_array_new)):
        X_protoset_cumuls[idx+diff] = mnemonics_array_new[idx]
    return X_protoset_cumuls
