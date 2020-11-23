import torch
import torch.nn as nn
from utils.misc import *

def process_inputs_fp(tg_model, free_model, inputs, fusion_mode=False, feature_mode=False):
    tg_model_group1 = [tg_model.conv1, tg_model.bn1, tg_model.relu, tg_model.layer1]
    tg_model_group1 = nn.Sequential(*tg_model_group1)
    tg_fp1 = tg_model_group1(inputs)
    fp1 = tg_fp1
    tg_model_group2 = tg_model.layer2
    tg_fp2 = tg_model_group2(fp1)
    fp2 = tg_fp2
    tg_model_group3 = [tg_model.layer3, tg_model.avgpool]
    tg_model_group3 = nn.Sequential(*tg_model_group3)
    tg_fp3 = tg_model_group3(fp2)
    fp3 = tg_fp3
    fp3 = fp3.view(fp3.size(0), -1)
    if feature_mode:
        return fp3
    else:
        outputs = tg_model.fc(fp3)
        feature = fp3
        return outputs, feature
