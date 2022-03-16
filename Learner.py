import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
import utils
import time
from datasets import convert_to_img
from datasets import preprocess
from datasets import postprocess
from torchvision.utils import save_image


class Trainer(object):

    def __init__(self, graph, optim, scheduler, trainingset, validset, args, cuda):

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


        self.num_epochs = args.num_epochs
        self.nll_gap = args.nll_gap
        self.inference_gap = args.inference_gap
        self.checkpoints_gap = args.checkpoints_gap
        self.save_gap = args.save_gap

        # device
        self.cuda = cuda

    def validate(self):
        print ("Start Validating")
        self.graph.eval()
        mean_loss = list()
        samples = list()
        with torch.no_grad():
            for i_batch, batch in enumerate(self.validset_loader):

                try:
                    x = batch["x"]
                    y = batch["y"]
                except Exception as e:
                    x, y = batch[0], batch[0]
                

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, True)


                # forward
                z, nll = self.graph(x,y)
                loss = torch.mean(nll)
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

                try:
                    x = batch["x"]
                    y = batch["y"]
                except Exception as e:
                    x, y = batch[0], batch[0]


                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()


                processed_y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, False)
                processed_x = preprocess(x, 1.0, 0.0, self.x_bins, False)

                # forward
                z, nll = self.graph(processed_x, processed_y)

                # loss
                loss = torch.mean(nll)
                mean_nll = mean_nll + loss.data

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)

                # step
                self.optim.step()

                currenttime = time.time()
                elapsed = currenttime - starttime
                print("Iteration: {}/{} \t Elapsed time: {:.2f} \t Loss:{:.5f}".format(self.global_step, total_its, elapsed, loss.data))

                if self.global_step % self.nll_gap == 0:
                    with open(os.path.join(self.log_dir, "NLL.txt"), "a") as nll_file:
                        nll_file.write(str(self.global_step) + " \t " + "{:.2f} \t {:.5f}".format(elapsed, loss.data) + "\n")



                # checkpoint
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    self.validate()

                    # samples
                    vis_batch = min(2, self.batch_size)
                    samples = []
                    for b in range(vis_batch):
                        samples.append([])

                    for i in range(0,5):
                        y_sample,_ = self.graph(x, y=None, reverse=True)
                        y_sample = postprocess(y_sample, self.label_scale, self.label_bias)
                        y_sample, _ = convert_to_img(y_sample)
                        for b in range(vis_batch):
                            samples[b].append(y_sample[b])

                    # inverse image
                    y_inverse,_ = self.graph(processed_x, y=z, reverse=True)
                    y_inverse = postprocess(y_inverse, self.label_scale, self.label_bias)
                    y_inverse, _ = convert_to_img(y_inverse)

                    # true label
                    y_true, _ = convert_to_img(y)


                    # save images
                    output = None
                    for b in range(0, vis_batch):
                        row = torch.cat((y_true[b], y_inverse[b]), dim=2)
                        for i in range(0, len(samples[b])):
                            row = torch.cat((row, samples[b][i]),dim=2)
                        if output is None:
                            output = row
                        else:
                            output = torch.cat((output, row), dim=1)

                    save_image(output, os.path.join(self.images_dir, "img-{}.png".format(self.global_step)))




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

            try:
                x = batch["x"]
                y = batch["y"]
            except Exception as e:
                x, y = batch[0], batch[0]

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
