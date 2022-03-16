import os
import torch
import torch.nn as nn
import datetime
import numpy as np
from torch.utils.data import DataLoader
import utils
import time
from datasets import convert_to_img
from datasets import preprocess
from datasets import postprocess
from torchvision.utils import save_image
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import predict_from_logits

class Trainer(object):

    def __init__(self, graph, adv_models, optim, scheduler, trainingset, validset, args, cuda):

        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, "log_" + args.name if args.name != '' else date)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.images_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.valid_samples_dir = os.path.join(self.log_dir, "valid_samples")
        if not os.path.exists(self.valid_samples_dir):
            os.makedirs(self.valid_samples_dir)



        # model
        self.graph = graph
        self.adv_models = adv_models
        self.optim = optim
        self.scheduler = scheduler

        # gradient bound
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.trainingset_loader = DataLoader(trainingset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True)

        self.validset_loader = DataLoader(validset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      drop_last=False)

        self.num_epochs = args.num_epochs
        self.global_step = args.num_steps
        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.x_bins = args.x_bins
        self.y_bins = args.y_bins
        self.margin = args.margin


        self.num_epochs = args.num_epochs
        self.nll_gap = args.nll_gap
        self.inference_gap = args.inference_gap
        self.checkpoints_gap = args.checkpoints_gap
        self.save_gap = args.save_gap
        self.target = args.target

        self.args = args

        # device
        self.cuda = cuda

    def adv_loss(self, y, label):
        loss = 0.
        for adv_model in self.adv_models:
#            loss = 0.
            logits = adv_model(y)

            if not self.target:
                one_hot= torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1,1)
                one_hot.scatter_(1, label, 1)
                one_hot = one_hot.bool()
                diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
            else:
                one_hot= torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1,1)
                one_hot.scatter_(1, label, 1)
                one_hot = one_hot.bool()
                diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
                margin = diff
            loss += margin.mean()
        loss /= len(self.adv_models)
            
        return loss 

    def augmentation(self, x, true_lab, no_adv=False):
        if self.args.adv_aug and (not no_adv):
            if self.args.adv_rand:
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]
                iter_num = np.random.randint(0, 20+1)

                if iter_num > 0:
                    adversary = LinfPGDAttack(
                        model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8./255,
                        nb_iter=iter_num, eps_iter=2./255, rand_init=True, clip_min=0.0,
                        clip_max=1.0, targeted=False)
                    with ctx_noparamgrad_and_eval(model_chosen):
                        x = adversary.perturb(x, None)
                else:
                    x = preprocess(x, 1.0, 0.0, self.x_bins, False)

            elif (not no_adv):
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]
            
                adversary = LinfPGDAttack(
                    model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8./255,
                    nb_iter=20, eps_iter=2./255, rand_init=True, clip_min=0.0,
                    clip_max=1.0, targeted=False)

                with ctx_noparamgrad_and_eval(model_chosen):
                    x = adversary.perturb(x, None)
        else:
            x = preprocess(x, 1.0, 0.0, self.x_bins, False)
        return x


    def validate(self):
        print ("Start Validating")
        self.graph.eval()
        mean_loss = list()
        samples = list()
        with torch.no_grad():
            for i_batch, batch in enumerate(self.validset_loader):
                x = batch["x"]
                y = batch["y"]
                label = batch["adv_lab"] if self.target else batch["true_lab"]

                if self.target:
                    label = torch.ones_like(batch["true_lab"]) * self.args.label

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    label = label.cuda()

                y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, True)


                # forward

                y, logdet = self.graph.sample(x)
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.tanh(y) * 8. / 255. + x, label)
                loss = loss_prob + loss_cls

                mean_loss.append(loss.data.cpu().item())


        # save loss
        mean = np.mean(mean_loss)
        with open(os.path.join(self.log_dir, "valid_NLL.txt"), "a") as nll_file:
            nll_file.write(str(self.global_step) + "\t" + "{:.5f}".format(mean) + "\n")
        print ("Finish Validating")
        self.graph.train()


    def train(self):

        self.graph.train()

        starttime = time.time()

        # run
        num_batchs = len(self.trainingset_loader)
        total_its = self.num_epochs * num_batchs
        for epoch in range(self.num_epochs):
            mean_nll = 0.0
            for _, batch in enumerate(self.trainingset_loader):
                self.optim.zero_grad()

                x = batch["x"]
                y = batch["y"]
                label = batch["adv_lab"] if self.target else batch["true_lab"]

                if self.target:
                    label = torch.ones_like(batch["true_lab"]) * self.args.label
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    label = label.cuda()

                processed_y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, False)
                processed_x = self.augmentation(x, label, epoch < self.args.adv_epoch)

                # forward
                y, logdet = self.graph.sample(processed_x)

                # loss
                if self.args.new_form:
                    loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.clamp(torch.clamp(y,  -8. / 255.,  8. / 255.) + x, 0, 1), label)
                    alpha = torch.exp(-loss_cls - loss_prob).detach()
                        
                    if self.args.only:
                        loss = loss_cls
                    else:
                        loss = self.args.Lambda * loss_prob + loss_cls 

                    loss = loss * alpha
                else:
                    if self.args.tanh:
                        loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.tanh(y) * 8. / 255. + x, label)
                    elif self.args.clamp:
                        loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.clamp(torch.clamp(y,  -8. / 255.,  8. / 255.) + x, 0, 1), label)
                    else:
                        loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(y / y.abs().max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1,1,1) * 8. / 255. + x, label)
                    
                    if self.args.only:
                        loss = loss_cls
                    else:
                        loss = self.args.Lambda * loss_prob + loss_cls 
                
                mean_nll = mean_nll + loss.data

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                if self.args.new_form:
                    
                    parameters = list(filter(lambda p: (p.grad is not None), self.graph.parameters()))

                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)

                    clip_coef = self.max_grad_norm / (total_norm + 1e-6)

                    for p in parameters:
                        p.grad.detach().mul_(clip_coef)

                # operate grad
                if self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)

                # step
                self.optim.step()

                currenttime = time.time()
                elapsed = currenttime - starttime
                print("Iteration: {}/{} \t Epoch: {}/{} \t Elapsed time: {:.2f} \t Loss:{:.5f} \t Loss_Prob:{:.5f} \t Loss_Cls:{:.5f}".format(self.global_step, total_its, epoch, self.num_epochs, elapsed, loss.data, loss_prob.data, loss_cls.data))
     
                if self.global_step % self.nll_gap == 0:
                    with open(os.path.join(self.log_dir, "NLL.txt"), "a") as nll_file:
                        nll_file.write(str(self.global_step) + " \t " + "{:.2f} \t {:.5f}".format(elapsed, loss.data) + "\n")



                # checkpoint
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    self.validate()
                  


                # save model
                if self.global_step % self.save_gap == 0 and self.global_step > 0:
                    utils.save_model(self.graph, self.optim, self.scheduler, self.checkpoints_dir, self.global_step)


                self.global_step = self.global_step + 1

            if self.scheduler is not None:
                self.scheduler.step()
            mean_nll = float(mean_nll / float(num_batchs))
            with open(os.path.join(self.log_dir, "Epoch_NLL.txt"), "a") as f:
                currenttime = time.time()
                elapsed = currenttime - starttime
                f.write("{} \t {:.2f}\t {:.5f}".format(epoch, elapsed, mean_nll) + "\n")




class Inferencer(object):

    def __init__(self, model, dataset, args, cuda):

        # set path and date
        self.out_root = args.out_root
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        # cuda
        self.cuda = cuda

        # model
        self.model = model


        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      drop_last=False)

        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.num_labels = args.num_labels


    def sampled_based_prediction(self, n_samples):
        metrics = []
        start = time.time()
        for i_batch, batch in enumerate(self.data_loader):
            print(f"Batch IDs: {i_batch}")

            x = batch["x"]
            y = batch["y"]

            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            sample_list = list()
            nll_list = list()
            for i in range(0, n_samples):

                print(f"Samples: {i}/{n_samples}")

                y_sample,_ = self.model(x, reverse=True)
                _, nll = self.model(x,y_sample)
                loss = torch.mean(nll)
                sample_list.append(y_sample)
                nll_list.append(loss.data.cpu().numpy())

            sample = torch.stack(sample_list)
            sample = torch.mean(sample, dim=0, keepdim=False)
            nll = np.mean(nll_list)


            sample = postprocess(sample, self.label_scale, self.label_bias)

            y_pred_imgs, y_pred_seg = convert_to_img(sample)
            y_true_imgs, y_true_seg = convert_to_img(y)



            # save trues and preds
            output = None
            for i in range(0, len(y_true_imgs)):
                true_img = y_true_imgs[i]
                pred_img = y_pred_imgs[i]
                row = torch.cat((x[i].cpu(), true_img, pred_img), dim=1)
                if output is None:
                    output = row
                else:
                    output = torch.cat((output,row), dim=2)
            save_image(output, os.path.join(self.out_root, "trues-{}.png".format(i_batch)))

            acc, acc_cls, mean_iu, fwavacc = utils.compute_accuracy(y_true_seg, y_pred_seg, self.num_labels)


            with open(os.path.join(self.out_root, "meta_list.txt"), "a") as meta_file:
                meta_file.write("NLL: {:.5f}".format(nll) + "\t")
                meta_file.write("acc: {:.8f}".format(acc) + "\t")
                meta_file.write("acc_cls: {:.8f}".format(acc_cls) + "\t")
                meta_file.write("mean_iu: {:.8f}".format(mean_iu) + "\t")
                meta_file.write("fwavacc: {:.8f}".format(fwavacc) + "\t")
                meta_file.write("\n")


            metrics.append([acc, acc_cls, mean_iu, fwavacc])
        mean_metrics = np.mean(metrics, axis=0)

        finish = time.time()
        elapsed = finish - start

        with open(os.path.join(self.out_root, "sum_meta.txt"), "w") as meta_file:
            meta_file.write("time:{:.2f}".format(elapsed) + "\t")
            meta_file.write("acc: {:.8f}".format(mean_metrics[0]) + "\t")
            meta_file.write("acc_cls: {:.8f}".format(mean_metrics[1]) + "\t")
            meta_file.write("mean_iu: {:.8f}".format(mean_metrics[2]) + "\t")
            meta_file.write("fwavacc: {:.8f}".format(mean_metrics[3]) + "\t")
