import os
import time
import datetime
import numpy as np

import nni
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib import utils


class KfoldToothTrainer:
    """
    Kfold Tooth Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.device = opt["device"]

        self.execute_dir = self.opt["execute_dir"]
        self.checkpoint_dir = self.opt["checkpoint_dir"]
        self.log_txt_path = self.opt["log_txt_path"]

        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_dsc = self.opt["best_dsc"]
        self.best_metrics = np.zeros((len(self.opt["metric_names"])))
        self.update_weight_freq = self.opt["update_weight_freq"]
        self.terminal_show_freq = self.opt["terminal_show_freq"]

        self.statistics_dict = self.init_statistics_dict()

    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.reset_statistics_dict()

            self.optimizer.zero_grad()

            self.train_epoch(epoch)

            self.valid_epoch(epoch)

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.statistics_dict["valid"]["DSC"] / self.statistics_dict["valid"]["count"])
            else:
                self.lr_scheduler.step()

            print(
                "[{}]  epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_hd:{:.6f}  valid_assd:{:.6f}  valid_iou:{:.6f}  valid_so:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch, self.end_epoch - 1,
                            self.optimizer.param_groups[0]['lr'],
                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["DSC"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["valid"]["HD"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["ASSD"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["IoU"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["SO"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["DSC"] / self.statistics_dict["valid"]["count"],
                            self.best_dsc))
            utils.pre_write_txt(
                "[{}]  epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_hd:{:.6f}  valid_assd:{:.6f}  valid_iou:{:.6f}  valid_so:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch, self.end_epoch - 1,
                            self.optimizer.param_groups[0]['lr'],
                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["DSC"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["valid"]["HD"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["ASSD"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["IoU"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["SO"] / self.statistics_dict["valid"]["count"],
                            self.statistics_dict["valid"]["DSC"] / self.statistics_dict["valid"]["count"],
                            self.best_dsc), self.log_txt_path)

        for i, metric_name in enumerate(self.opt["metric_names"]):
            self.opt["metric_results_per_fold"][metric_name].append(self.best_metrics[i])

    def train_epoch(self, epoch):

        self.model.train()

        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):

            input_tensor, target = input_tensor.to(self.device), target.to(self.device)

            output = self.model(input_tensor)
            dice_loss = self.loss_function(output, target)

            scaled_dice_loss = dice_loss / self.update_weight_freq
            scaled_dice_loss.backward()

            if (batch_idx + 1) % self.update_weight_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.calculate_metric_and_update_statistcs(output.cpu().float(), target.cpu().float(), len(target), dice_loss.cpu(), mode="train")

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                print("[{}]  epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}"
                      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              epoch, self.end_epoch - 1,
                              batch_idx, len(self.train_data_loader) - 1,
                              self.optimizer.param_groups[0]['lr'],
                              self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["DSC"] / self.statistics_dict["train"]["count"]))
                utils.pre_write_txt("[{}]  epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}"
                                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            epoch, self.end_epoch - 1,
                                            batch_idx, len(self.train_data_loader) - 1,
                                            self.optimizer.param_groups[0]['lr'],
                                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                            self.statistics_dict["train"]["DSC"] / self.statistics_dict["train"]["count"]),
                                    self.log_txt_path)

    def valid_epoch(self, epoch):

        self.model.eval()

        with torch.no_grad():

            for batch_idx, (input_tensor, target) in enumerate(self.valid_data_loader):

                input_tensor, target = input_tensor.to(self.device), target.to(self.device)

                output = self.split_forward(input_tensor, self.model)

                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target), mode="valid")

            cur_dsc = self.statistics_dict["valid"]["DSC"] / self.statistics_dict["valid"]["count"]

            if cur_dsc > self.best_dsc:
                self.best_dsc = cur_dsc
                self.best_metrics = np.array([
                    self.statistics_dict["valid"]["HD"] / self.statistics_dict["valid"]["count"],
                    self.statistics_dict["valid"]["ASSD"] / self.statistics_dict["valid"]["count"],
                    self.statistics_dict["valid"]["IoU"] / self.statistics_dict["valid"]["count"],
                    self.statistics_dict["valid"]["SO"] / self.statistics_dict["valid"]["count"],
                    self.statistics_dict["valid"]["DSC"] / self.statistics_dict["valid"]["count"]
                ])

            self.save(epoch, cur_dsc, self.best_dsc, type="latest")

    def split_forward(self, image, model):
        ori_shape = image.size()[2:]
        output = torch.zeros((image.size()[0], self.opt["classes"], *ori_shape), device=image.device)
        slice_shape = self.opt["crop_size"]
        stride = self.opt["crop_stride"]

        for shape0_start in range(0, ori_shape[0], stride[0]):
            shape0_end = shape0_start + slice_shape[0]
            start0 = shape0_start
            end0 = shape0_end
            if shape0_end >= ori_shape[0]:
                end0 = ori_shape[0]
                start0 = end0 - slice_shape[0]

            for shape1_start in range(0, ori_shape[1], stride[1]):
                shape1_end = shape1_start + slice_shape[1]
                start1 = shape1_start
                end1 = shape1_end
                if shape1_end >= ori_shape[1]:
                    end1 = ori_shape[1]
                    start1 = end1 - slice_shape[1]

                for shape2_start in range(0, ori_shape[2], stride[2]):
                    shape2_end = shape2_start + slice_shape[2]
                    start2 = shape2_start
                    end2 = shape2_end
                    if shape2_end >= ori_shape[2]:
                        end2 = ori_shape[2]
                        start2 = end2 - slice_shape[2]

                    slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                    slice_predict = model(slice_tensor.to(image.device))
                    output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict

                    if shape2_end >= ori_shape[2]:
                        break

                if shape1_end >= ori_shape[1]:
                    break

            if shape0_end >= ori_shape[0]:
                break

        return output

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        self.statistics_dict[mode]["count"] += cur_batch_size
        if mode == "train":
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size
        for i, metric_name in enumerate(self.opt["metric_names"]):
            if (mode == "train" and metric_name == "DSC") or (mode == "valid"):
                per_class_metric = self.metric[i](output, target)
                self.statistics_dict[mode][metric_name] += (torch.sum(per_class_metric) / 2.0).item() * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            "train": {},
            "valid": {}
        }
        statistics_dict["train"]["loss"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name] = 0.0
            statistics_dict["valid"][metric_name] = 0.0
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0
        return statistics_dict

    def reset_statistics_dict(self):
        for phase in ["train", "valid"]:
            self.statistics_dict[phase]["count"] = 0
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name] = 0.0

    def save(self, epoch, metric, best_metric, type="normal"):
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metrics": self.best_metrics,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fold": self.opt["current_fold"],
            "metric_results_per_fold": self.opt["metric_results_per_fold"]
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        if self.opt["resume"] is not None:
            if self.opt["pretrain"] is None:
                raise RuntimeError("Need pretrain path when resume training")

            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.start_epoch = resume_state_dict["epoch"] + 1
            self.best_dsc = resume_state_dict["best_metric"]
            self.best_metrics = resume_state_dict["best_metrics"]
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            model_state_dict = self.model.state_dict()
            load_count = 0
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            self.model.load_state_dict(model_state_dict, strict=True)
            print("{:.2f}% of parameters load weights success".format(100 * load_count / len(model_state_dict)))
            utils.pre_write_txt("{:.2f}% of parameters load weights success".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
        else:
            if self.opt["pretrain"] is not None:
                pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                model_state_dict = self.model.state_dict()
                load_count = 0
                for param_name in model_state_dict.keys():
                    if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                        model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                        load_count += 1
                self.model.load_state_dict(model_state_dict, strict=True)
                print("{:.2f}% of parameters load weights success".format(100 * load_count / len(model_state_dict)))
                utils.pre_write_txt("{:.2f}% of parameters load weights success".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
