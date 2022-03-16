import argparse
import torchvision.models as models
import os
import json

import torch
import Learner
import datasets
import utils
from CGlowModel import CondGlowModel
import os
from surro_models import resnet_preact, resnet, pyramidnet, wrn, vgg, densenet
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import preprocess
from datasets import postprocess
import ast

import cma
from cma.fitness_transformations import Function as FFun
import cma.fitness_models as fm

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='train c-Glow')

# input output path
parser.add_argument("-d", "--dataset_name", type=str, default="horse")
parser.add_argument("-r", "--dataset_root", type=str, default="")

# log root
parser.add_argument("--log_root", type=str, default="")

# C-Glow parameters
parser.add_argument("--x_size", type=tuple, default=(3,32,32))
parser.add_argument("--y_size", type=tuple, default=(3,32,32))
parser.add_argument("--x_hidden_channels", type=int, default=128)
parser.add_argument("--x_hidden_size", type=int, default=64)
parser.add_argument("--y_hidden_channels", type=int, default=256)
parser.add_argument("-K", "--flow_depth", type=int, default=8)
parser.add_argument("-L", "--num_levels", type=int, default=3)
parser.add_argument("--learn_top", type=ast.literal_eval, default=False)


# Dataset preprocess parameters
parser.add_argument("--label_scale", type=float, default=1)
parser.add_argument("--label_bias", type=float, default=0.)
parser.add_argument("--x_bins", type=float, default=256.0)
parser.add_argument("--y_bins", type=float, default=2.0)


# Optimizer parameters
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--betas", type=tuple, default=(0.9,0.9999))
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--regularizer", type=float, default=0.0)
parser.add_argument("--num_steps", type=int, default=0)

# Trainer parameters
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--down_sample", type=int, default=1)
parser.add_argument("--max_grad_clip", type=float, default=5)
parser.add_argument("--max_grad_norm", type=float, default=0)
parser.add_argument("--checkpoints_gap", type=int, default=1000)
parser.add_argument("--nll_gap", type=int, default=1)
parser.add_argument("--inference_gap", type=int, default=1000)
parser.add_argument("--save_gap", type=int, default=1000)

parser.add_argument("--target", type=ast.literal_eval, default=False)
parser.add_argument("--tanh", type=ast.literal_eval, default=False)
parser.add_argument("--only", type=ast.literal_eval, default=False)
parser.add_argument("--rand", type=ast.literal_eval, default=False)

# model path
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--name", type=str, default="")

parser.add_argument("--num_classes", type=int, default=0)
parser.add_argument("--class_size", type=int, default=-1)

parser.add_argument("--targetn", type=str, default="resnet")
parser.add_argument('--target-type', default='random', type=str, choices=['random', 'least_likely', 'most_likely', 'median'],
                        help='how to choose target class for targeted attack, could be random or least_likely')

args = parser.parse_args()
args.batch_size = 1 
cuda = torch.cuda.is_available()


class Function(nn.Module):

    def __init__(self, model, batch_size=256, margin=0, nlabels=10, target=False):
        super(Function, self).__init__()
        self.model = model
        self.margin = margin
        self.target = target
        self.batch_size = batch_size
        self.current_counts = 0
        self.counts = []
        self.nlabels = nlabels

    def _loss(self, logits, label):
        if not self.target:
            if label == 0:
                logits_cat = logits[:,(label+1):]
            elif label == logits.size()[1] - 1:
                logits_cat = logits[:, :label]
            else:
                logits_cat = torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1)
            diff = logits[:,label] - torch.max(logits_cat, dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin

    def forward(self, images, label):
        if len(images.size())==3:
            images = images.unsqueeze(0)
        n = len(images)
        device = images.device
        k = 0
        loss = torch.zeros(n, dtype=torch.float32, device=device)
        logits = torch.zeros((n, self.nlabels), dtype=torch.float32, device=device)

        while k < n:
            start = k
            end = min(k + self.batch_size, n)
            logits[start:end] = self.model(images[start:end])
            loss[start:end] = self._loss(logits, label)
            k = end
        self.current_counts += n

        return logits, loss

    def new_counter(self):
        self.counts.append(self.current_counts)
        self.current_counts = 0

    def get_average(self, iter=1000):
        counts = np.array(self.counts)
        print ('seeL', counts)
        return np.mean(counts[counts<iter])

    def get_median(self, iter=1000):
        counts = np.array(self.counts)
        return np.median(counts[counts<iter])

    def get_first_success(self):
        counts = np.array(self.counts)
        return np.mean(counts == 1)






valid_set = datasets.cifar10(args.dataset_root, (args.y_size[1], args.y_size[2]), args.y_size[0], portion="valid")


prob_model = CondGlowModel(args)
state = utils.load_state(args.model_path, cuda)

prob_model.load_state_dict(state["model"])
if cuda:
    prob_model = prob_model.cuda()
prob_model.eval()


def load_prob():
    prob_model = CondGlowModel(args)
    state = utils.load_state(args.model_path, cuda)
    if not args.rand:
        prob_model.load_state_dict(state["model"])
    if cuda:
        prob_model = prob_model.cuda()
    prob_model.eval()
    return prob_model

model_name = args.targetn

def load_net(model_name):
    if model_name == "pyramidnet":
        TRAINED_MODEL_PATH = './classification_models/pyramidnet_basic_110_84/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = pyramidnet.Network(json.load(fr)['model_config'])
            
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'resnet_adv_4':
        TRAINED_MODEL_PATH = './classification_models/resnet_adv_4/cifar-10_linf/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
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

    from advertorch.utils import NormalizeByChannelMeanStd

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    normalize = NormalizeByChannelMeanStd(
            mean=mean.tolist(), std=std.tolist())
    net = nn.Sequential(
        normalize,
        pretrained_model
    )

    if cuda:
        net = net.cuda()
    net.eval()
    return net

net = load_net(model_name)
ref_names = ["pyramidnet", "resnet", "vgg", "dense"]
net_ref = []
for mn in ref_names:
    if mn != model_name:
        net_ref.append(load_net(mn))

valid_set = datasets.cifar10(args.dataset_root, (args.y_size[1], args.y_size[2]), args.y_size[0], portion="valid")
validset_loader = DataLoader(valid_set,
              batch_size=args.batch_size,
              shuffle=False,
              drop_last=False)



global history

import sys

target = args.target

success, cnt = 0., 0.


F = Function(net, 256, 5.0, 10, target)

count_success = 0
count_total = 0

for i, batch in enumerate(validset_loader):


    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    prob_model = load_prob()
    prob_model.eval()

    images = batch["x"]
    y = batch["y"]
    labels = batch["true_lab"] if target else batch["true_lab"]
    print ('Norm:', i, images.norm())    

    images = images.cuda()
    y = y.cuda()
    labels = int(labels)
    logits = net(images)

    correct = torch.argmax(logits, dim=1) == labels
    

    if correct:
        history = []

        torch.cuda.empty_cache()

        if target:
           if args.target_type == 'random':
                labels = torch.randint(low=0, high=logits.shape[1], size=labels.shape).long().to(device)
            elif args.target_type == 'least_likely':
                labels = int(logits.argmin(dim=1))
            elif args.target_type == 'most_likely':
                labels = int(torch.argsort(logits, dim=1,descending=True)[:,1])
       elif args.target_type == 'median':
                labels = int(torch.argsort(logits, dim=1,descending=True)[:,4])

            else:
                raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

        
        def pred_interface(prob_model, image, latent_vec):
            def pred(latent):
                perturbation, _ = prob_model.flow.decode(image.unsqueeze(0), latent, zs=latent_vec)
                original_img = image
                perturbed_img = torch.clamp(original_img + torch.sign(perturbation) *  8./255, 0, 1)
                logit = net(perturbed_img)
               return logit
            return pred

        def pred_ref_interface(prob_model, image, latent_vec):
            def pred(latent):
               perturbation, _ = prob_model.flow.decode(image.unsqueeze(0), latent, zs=latent_vec)
                original_img = image
                perturbed_img = torch.clamp(original_img + torch.sign(perturbation) *  8./255, 0, 1)
                logit = 0
                for net_r in net_ref:
                    logit += net_r(perturbed_img)
                logit /= len(net_ref)
                return logit
            return pred

        def criterion_interface(target, labels):

            def criterion(logits):
                threshold = 20.0
                if not target:
                    if labels == 0:
                        logits_cat = logits[:,(labels+1):]
                    elif labels == logits.size()[1] - 1:
                        logits_cat = logits[:, :labels]
                    else:
                        logits_cat = torch.cat((logits[:, :labels],logits[:,(labels+1):]), dim=1)
                    diff = logits[:,labels] - torch.max(logits_cat, dim=1)[0]
                    margin = torch.nn.functional.relu(diff + threshold, True) - threshold
                else:
                    diff = torch.max(torch.cat((logits[:, :labels],logits[:,(labels+1):]), dim=1), dim=1)[0] - logits[:, labels]
                    margin = torch.nn.functional.relu(diff + threshold, True) - threshold
                return margin.item()

            return criterion

        def compound_interface(prob_model, image, latent_vec, target, labels, temp_size):
            pred = pred_interface(prob_model, image, latent_vec)
            criterion = criterion_interface(target, labels)

            def compound(latent):
                latent = latent.reshape(temp_size)
                latent = torch.FloatTensor(latent).unsqueeze(0).cuda()
                return criterion(pred(latent))
            return compound

        def compound_ref_interface(prob_model, image, latent_vec, target, labels, temp_size):
            pred = pred_ref_interface(prob_model, image, latent_vec)
            criterion = criterion_interface(target, labels)

            def compound(latent):
                latent = latent.reshape(temp_size)
                latent = torch.FloatTensor(latent).unsqueeze(0).cuda()
                return criterion(pred(latent))
            return compound


        def init(image, prob_model):
            init_pert, p = prob_model.decode(image.unsqueeze(0), return_prob=True, no_norm=True)
            init_pert = init_pert.squeeze(0)

            latent, pp, latent_vec = prob_model.flow.encode(image.unsqueeze(0), init_pert.unsqueeze(0), return_z=True)
            latent_base = latent.clone()
            latent_vec = [lat.detach() for lat in latent_vec]

            return latent_base, latent_vec
        
        latent, latent_vec = init(images[0], prob_model)
        
        es = cma.CMAEvolutionStrategy(latent.squeeze(0).cpu().data.numpy().reshape(-1), 1e5, {'seed':666, 'maxfevals':10000, 'popsize':20, 'ftarget':0})
        compound = compound_interface(prob_model, images[0], latent_vec, target, labels, latent.squeeze(0).cpu().data.numpy().shape)
        compound_ref = compound_ref_interface(prob_model, images[0], latent_vec, target, labels, latent.squeeze(0).cpu().data.numpy().shape)



        score0 = compound(latent.squeeze(0).cpu().data.numpy().reshape(-1))
        if score0 < 0:
            query_cnt, success = 1, True
        else:
            score0 = -1
            query_cnt = 1
            cntn = 0
            while not es.stop() and es.best.f > 0:
                X = es.ask()  # get list of new solutions

                query_cnt += len(X)

                fit = [compound(x) for x in X]  
                es.tell(X, fit)  # feed values
                print ('Iteration:', cntn, 'ES best f: ', es.best.f)
                cntn += 1
                sys.stdout.flush()
                score0 = max(score0, es.best.f)
                
            success = (list(es.stop().keys())[0] == 'ftarget')

        print (es.best.f, es.stop())
        F.current_counts = query_cnt

        count_success += int(success)
        count_total += int(correct)
        print("image: {} eval_count: {} success: {} average_count: {} median_count: {} success_at_once: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(6000), F.get_median(6000), F.get_first_success(), float(count_success) / float(count_total)))
        sys.stdout.flush()
        F.new_counter()
    if count_total > 1000:
        break

success_rate = float(count_success) / float(count_total)
print("success rate {}".format(success_rate))
print("average eval count {}".format(F.get_average(10000)))
print("median eval count {}".format(F.get_median(10000)))
