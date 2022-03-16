import os
import torch
import numpy as np
from surro_models import resnet_preact, resnet, pyramidnet, wrn, vgg, densenet
import json
from torch import nn
import torchvision.models as models

def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def save_model(model, optim, scheduler, dir, iteration):
    path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
    state = {}
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None

    torch.save(state, path)


def load_state(path, cuda):
    if cuda:
        print ("load to gpu")
        state = torch.load(path)
    else:
        print ("load to cpu")
        state = torch.load(path, map_location=lambda storage, loc: storage)

    return state


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_accuracy(label_trues, label_preds, n_class):

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_adv(model_list = ['pyramidnet']):
    nets = []

    for model_name in model_list:
        if model_name == "pyramidnet":
            TRAINED_MODEL_PATH = './classification_models/pyramidnet_basic_110_84/00/'
            filename = 'model_best_state.pth'
            with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
                pretrained_model = pyramidnet.Network(json.load(fr)['model_config'])
                pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
        elif model_name == 'resnet':
            TRAINED_MODEL_PATH = './classification_models/resnet_basic_110/00/'
            filename = 'model_best_state.pth'
            with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
                pretrained_model = resnet.Network(json.load(fr)['model_config'])
                pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
        elif model_name == 'wrn':
            TRAINED_MODEL_PATH = './classification_models/wrn_28_10/00/'
            filename = 'model_best_state.pth'
            with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
                pretrained_model = wrn.Network(json.load(fr)['model_config'])
                pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
        elif model_name == 'vgg':
            TRAINED_MODEL_PATH = './classification_models/vgg_15_BN_64/00/'
            filename = 'model_best_state.pth'
            with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
                pretrained_model = vgg.Network(json.load(fr)['model_config'])
                pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
        elif model_name == 'dense':
            TRAINED_MODEL_PATH = './classification_models/densenet_BC_100_12/00/'
            filename = 'model_best_state.pth'
            with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
                pretrained_model = densenet.Network(json.load(fr)['model_config'])
                pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])


        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        from advertorch.utils import NormalizeByChannelMeanStd
        
        normalize = NormalizeByChannelMeanStd(
                mean=mean.tolist(), std=std.tolist())
        net = nn.Sequential(
            normalize,
            pretrained_model
        )
        nets.append(net)

    for i in range(len(nets)):
        nets[i] = nets[i].cuda()
        nets[i].eval()
        

    return nets


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        # print (size)
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x

def load_adv_imagenet(model_list = ['VGG16', 'Resnet18', 'Googlenet']):
    nets = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for model_name in model_list:
        print(model_name)
        if model_name == "VGG16":
            pretrained_model = models.vgg16_bn(pretrained=True)
        elif model_name == 'Resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'Squeezenet':
            pretrained_model = models.squeezenet1_1(pretrained=True)
        elif model_name == 'Googlenet':
            pretrained_model = models.googlenet(pretrained=True)
        elif model_name == 'Adv_Denoise_Resnet152':
            pretrained_model = resnet152_denoise()
            loaded_state_dict = torch.load(os.path.join('weight', model_name+".pytorch"))
            pretrained_model.load_state_dict(loaded_state_dict)

        net = nn.Sequential(
            Normalize(mean, std),
            pretrained_model
        )
        nets.append(net)

    for i in range(len(nets)):
        nets[i] = nets[i].cuda()
        nets[i].eval()
        
    return nets