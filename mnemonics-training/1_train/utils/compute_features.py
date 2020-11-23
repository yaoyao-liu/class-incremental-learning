import torch
import numpy as np
from torchvision import models
from utils.misc import *
from utils.process_fp import process_inputs_fp

def compute_features(tg_model, free_model, tg_feature_model, is_start_iteration, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()
    tg_model.eval()
    if free_model is not None:
        free_model.eval()
    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            if is_start_iteration:
                the_feature = tg_feature_model(inputs)
            else:
                the_feature = process_inputs_fp(tg_model, free_model, inputs, feature_mode=True)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(the_feature)
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
