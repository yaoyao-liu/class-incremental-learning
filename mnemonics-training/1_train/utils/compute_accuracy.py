import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from scipy.spatial.distance import cdist
from utils.misc import *
from utils.process_fp import process_inputs_fp

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def compute_accuracy(tg_model, free_model, tg_feature_model, class_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=False, fast_fc=None, scale=None, print_info=True, device=None, maml_lr=0.1, maml_epoch=50):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()
    tg_model.eval()
    if free_model is not None:
        free_model.eval()
    if fast_fc is None:
        transform_proto = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
        protoset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_proto)
        X_protoset_array = np.array(X_protoset_cumuls).astype('uint8')
        protoset.test_data = X_protoset_array.reshape(-1, X_protoset_array.shape[2], X_protoset_array.shape[3], X_protoset_array.shape[4])
        Y_protoset_cumuls = np.array(Y_protoset_cumuls).reshape(-1)
        map_Y_protoset_cumuls = map_labels(order_list, Y_protoset_cumuls)
        protoset.test_labels = map_Y_protoset_cumuls
        protoloader = torch.utils.data.DataLoader(protoset, batch_size=128, shuffle=True, num_workers=2)  

        fast_fc = torch.from_numpy(np.float32(class_means[:,:,0].T)).to(device)
        fast_fc.requires_grad=True

        epoch_num = maml_epoch
        for epoch_idx in range(epoch_num):
            for the_inputs, the_targets in protoloader: 
                the_inputs, the_targets = the_inputs.to(device), the_targets.to(device)
                the_features = tg_feature_model(the_inputs)
                the_logits = F.linear(F.normalize(torch.squeeze(the_features), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
                the_loss = F.cross_entropy(the_logits, the_targets)
                the_grad = torch.autograd.grad(the_loss, fast_fc)
                fast_fc = fast_fc - maml_lr * the_grad[0]
    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    correct_maml = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            if is_start_iteration:
                outputs = tg_model(inputs)
            else:
                outputs, outputs_feature = process_inputs_fp(tg_model, free_model, inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            if is_start_iteration:
                outputs_feature = np.squeeze(tg_feature_model(inputs))
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            the_logits = F.linear(F.normalize(torch.squeeze(outputs_feature), p=2,dim=1), F.normalize(fast_fc, p=2, dim=1))
            _, predicted_maml = the_logits.max(1)
            correct_maml += predicted_maml.eq(targets).sum().item()
    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total
    maml_acc = 100.*correct_maml/total
    if print_info:
        print("  Accuracy for LwF    :\t\t{:.2f} %".format(cnn_acc))
        print("  Accuracy for iCaRL  :\t\t{:.2f} %".format(icarl_acc))
        print("  The above results are the accuracy for the current phase.") 
        print("  For the average accuracy, you need to record the results for all phases and calculate the average value.") 
    return [cnn_acc, icarl_acc, ncm_acc, maml_acc], fast_fc
