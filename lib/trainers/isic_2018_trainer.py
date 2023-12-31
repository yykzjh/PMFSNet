import os
import time
import numpy as np
import datetime

import nni
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib import utils


class ISIC2018Trainer:
    """
    Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        # 传入的参数
        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.device = opt["device"]

        if not self.opt["optimize_params"]:
            # 创建训练执行目录和文件
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"], utils.datestr() + "_" + opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                utils.make_dirs(self.checkpoint_dir)
                utils.make_dirs(self.tensorboard_dir)
            # 使用的模型、优化器、学习率调度器记录到日志文件
            utils.pre_write_txt("初始化模型:{}、优化器:{}和学习率调整器:{}".format(self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]), self.log_txt_path)

        # 训练时需要用到的参数
        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_metric = opt["best_metric"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

        # 训练的中间统计信息
        self.statistics_dict = self.init_statistics_dict()

    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            # 重置中间统计信息字典
            self.reset_statistics_dict()

            # 梯度清0
            self.optimizer.zero_grad()

            # 当前epoch的训练阶段
            self.train_epoch(epoch)

            # 当前epoch的验证阶段
            self.valid_epoch(epoch)

            # 计算训练集上的IoU和mIoU
            train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
            train_class_IoU = np.nan_to_num(train_class_IoU)
            # 计算验证集上的IoU和mIoU
            valid_class_IoU = self.statistics_dict["valid"]["total_area_intersect"] / self.statistics_dict["valid"]["total_area_union"]
            valid_class_IoU = np.nan_to_num(valid_class_IoU)
            # 计算验证集上的dsc
            valid_dsc = self.statistics_dict["valid"]["DSC_sum"] / self.statistics_dict["valid"]["count"]
            # 计算验证集上的JI
            valid_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]
            # 计算验证集上的ACC
            valid_ACC = self.statistics_dict["valid"]["ACC_sum"] / self.statistics_dict["valid"]["count"]

            # 更新学习率
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(valid_JI)
            else:
                self.lr_scheduler.step()

            # epoch结束总的输出一下结果
            print(
                "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC:{:.6f}  train_JI:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f}"
                .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch, self.end_epoch - 1,
                        self.optimizer.param_groups[0]['lr'],
                        self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                        self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                        train_class_IoU[1],
                        self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                        self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"],
                        valid_dsc,
                        valid_class_IoU[1],
                        valid_ACC,
                        valid_JI,
                        self.best_metric))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt(
                    "[{}]  epoch:[{:05d}/{:05d}]  lr:{:.6f}  train_loss:{:.6f}  train_DSC:{:.6f}  train_IoU:{:.6f}  train_ACC:{:.6f}  train_JI:{:.6f}  valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}  best_JI:{:.6f}"
                    .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch, self.end_epoch - 1,
                            self.optimizer.param_groups[0]['lr'],
                            self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                            train_class_IoU[1],
                            self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                            self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"],
                            valid_dsc,
                            valid_class_IoU[1],
                            valid_ACC,
                            valid_JI,
                            self.best_metric), self.log_txt_path)

            if self.opt["optimize_params"]:
                # 向nni上报每个epoch验证集的mIoU作为中间指标
                nni.report_intermediate_result(valid_JI)

        if self.opt["optimize_params"]:
            # 将在验证集上最优的mIoU作为最终上报指标
            nni.report_final_result(self.best_metric)

    def train_epoch(self, epoch):

        # 训练
        self.model.train()

        # 遍历数据集的batch
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):

            # 将输入图像和标注图像都移动到指定设备上
            input_tensor, target = input_tensor.to(self.device), target.to(self.device)
            # 前向传播
            output = self.model(input_tensor)
            # 计算损失值
            dice_loss = self.loss_function(output, target)
            # 反向传播
            dice_loss.backward()
            # 更新参数
            self.optimizer.step()
            # 梯度清0
            self.optimizer.zero_grad()

            # 计算各评价指标并更新中间统计信息
            self.calculate_metric_and_update_statistcs(output.cpu().float(), target.cpu().float(), len(target), dice_loss.cpu(), mode="train")

            # 判断满不满足打印信息或者画图表的周期
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                # 计算IoU和mIoU
                train_class_IoU = self.statistics_dict["train"]["total_area_intersect"] / self.statistics_dict["train"]["total_area_union"]
                train_class_IoU = np.nan_to_num(train_class_IoU)
                # 打印
                print("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              epoch, self.end_epoch - 1,
                              batch_idx + 1, len(self.train_data_loader),
                              self.optimizer.param_groups[0]['lr'],
                              self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                              train_class_IoU[1],
                              self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("[{}]  epoch:[{:05d}/{:05d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}  IoU:{:.6f}  ACC:{:.6f}  JI:{:.6f}"
                                        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                epoch, self.end_epoch - 1,
                                                batch_idx + 1, len(self.train_data_loader),
                                                self.optimizer.param_groups[0]['lr'],
                                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["DSC_sum"] / self.statistics_dict["train"]["count"],
                                                train_class_IoU[1],
                                                self.statistics_dict["train"]["ACC_sum"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["JI_sum"] / self.statistics_dict["train"]["count"]),
                                        self.log_txt_path)

    def valid_epoch(self, epoch):

        # 验证集测试
        self.model.eval()

        # 测试时不保存计算图的梯度中间结果，加快速度，节省空间
        with torch.no_grad():

            # 遍历验证集的batch，默认一个batch一张图像
            for batch_idx, (input_tensor, target) in enumerate(self.valid_data_loader):
                # 将输入图像和标注图像都移动到指定设备上
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(input_tensor)

                # 计算各评价指标并更新中间统计信息
                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target), mode="valid")

            # 计算验证集上的JI
            cur_JI = self.statistics_dict["valid"]["JI_sum"] / self.statistics_dict["valid"]["count"]

            # 按照一定周期固定保存模型和训练状态部分
            if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0:
                self.save(epoch, cur_JI, self.best_metric, type="normal")
            if not self.opt["optimize_params"]:
                # 每次都保存最新的latest
                self.save(epoch, cur_JI, self.best_metric, type="latest")
            # 与最优结果进行比较，保存最优的模型
            if cur_JI > self.best_metric:
                self.best_metric = cur_JI
                if not self.opt["optimize_params"]:
                    self.save(epoch, cur_JI, self.best_metric, type="best")

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        """
        计算评价指标并更新中间统计信息字典
        Args:
            output: 网络输出的张量
            target: 目标标注张量
            loss: 损失值
            cur_batch_size: 当前batch的大小
            mode: "train"|"valid"

        Returns:
        """
        # 计算出现的类别mask
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        # 更新总样本数
        self.statistics_dict[mode]["count"] += cur_batch_size
        # 更新各类别计数
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict[mode]["class_count"][class_name] += cur_batch_size
        # 如果是训练阶段要更新损失值
        if mode == "train":
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size
        # 更新各评价指标
        for metric_name, metric_func in self.metric.items():
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict[mode]["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict[mode]["total_area_union"] += area_union.numpy()
            elif metric_name == "ACC":
                batch_mean_ACC = metric_func(output, target)
                self.statistics_dict[mode]["ACC_sum"] += batch_mean_ACC * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target)
                self.statistics_dict[mode]["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target)
                self.statistics_dict[mode]["DSC_sum"] += batch_mean_DSC * cur_batch_size
            else:
                # 计算当前评价指标在各类别上的数值
                per_class_metric = metric_func(output, target)
                # 只需要mask部分
                per_class_metric = per_class_metric * mask
                # 更新平均评价指标
                self.statistics_dict[mode][metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                # 更新各类别的各评价指标
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[mode][metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def init_statistics_dict(self):
        # 初始化所有评价指标在所有类别上的统计数据
        statistics_dict = {
            "train": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            },
            "valid": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            }
        }
        # 初始化IoU的交集统计直方图和并集统计直方图
        statistics_dict["train"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["train"]["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["valid"]["total_area_union"] = np.zeros((self.opt["classes"],))
        # 初始化JI的累计和
        statistics_dict["train"]["JI_sum"] = 0.0
        statistics_dict["valid"]["JI_sum"] = 0.0
        # 初始化ACC的累计和
        statistics_dict["train"]["ACC_sum"] = 0.0
        statistics_dict["valid"]["ACC_sum"] = 0.0
        # 初始化DSC的累计和
        statistics_dict["train"]["DSC_sum"] = 0.0
        statistics_dict["valid"]["DSC_sum"] = 0.0
        # 初始化所有评价指标在所有类别上的平均值
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name]["avg"] = 0.0
            statistics_dict["valid"][metric_name]["avg"] = 0.0
        # 初始化损失值
        statistics_dict["train"]["loss"] = 0.0
        # 初始化各类别计数
        statistics_dict["train"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["valid"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        # 初始化所有样本计数
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        for phase in ["train", "valid"]:
            # 重置所有样本计数
            self.statistics_dict[phase]["count"] = 0
            # 重置IoU的交集统计直方图和并集统计直方图
            self.statistics_dict[phase]["total_area_intersect"] = np.zeros((self.opt["classes"],))
            self.statistics_dict[phase]["total_area_union"] = np.zeros((self.opt["classes"],))
            # 初始化JI的累计和
            self.statistics_dict[phase]["JI_sum"] = 0.0
            # 初始化ACC的累计和
            self.statistics_dict[phase]["ACC_sum"] = 0.0
            # 初始化DSC的累计和
            self.statistics_dict[phase]["DSC_sum"] = 0.0
            # 重置各类别计数
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[phase]["class_count"][class_name] = 0
            # 重置损失值
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
            # 重置平均评价指标
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name]["avg"] = 0.0
                # 重置各类别的各评价指标
                for _, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[phase][metric_name][class_name] = 0.0

    def save(self, epoch, metric, best_metric, type="normal"):
        """
        保存当前训练状态和模型参数
        Args:
            epoch: 当前迭代数
            metric: 当前评价指标
            best_metric: 当前最优的评价指标
            type: 存储类型，可选："normal"|"best"|"latest"

        Returns:
        """
        # 保存当前训练状态的状态字典
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        # 保存模型参数
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        """
        根据参数判断是否需要加载训练状态或者模型预训练参数
        Returns:
        """
        if self.opt["resume"] is not None:  # 如果需要继续训练
            # 此时预训练模型参数必须指定，否则抛出错误
            if self.opt["pretrain"] is None:
                raise RuntimeError("继续训练必须指定预训练模型参数")

            # 加载训练状态字典
            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            # 加载当前epoch
            self.start_epoch = resume_state_dict["epoch"] + 1
            # 加载当前最优评价指标
            self.best_metric = resume_state_dict["best_metric"]
            # 加载优化器参数
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            # 加载学习率调度器参数
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            # 加载模型参数字典
            pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            # 获取模型参数字典
            model_state_dict = self.model.state_dict()
            # 遍历模型参数
            load_count = 0  # 成功加载参数计数
            for param_name in model_state_dict.keys():
                # 判断当前模型参数是否在预训练参数中
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            # 严格加载模型参数
            self.model.load_state_dict(model_state_dict, strict=True)
            # 输出权重参数加载率
            print("{:.2f}%的模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)))
            # 将预训练权重加载情况记录到日志文件
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("{:.2f}%的模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
        else:  # 如果不需要继续训练
            # 有可能需要加载模型的预训练参数
            if self.opt["pretrain"] is not None:
                # 加载模型参数字典
                pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                # 获取模型参数字典
                model_state_dict = self.model.state_dict()
                # 遍历模型参数
                load_count = 0  # 成功加载参数计数
                for param_name in model_state_dict.keys():
                    # 判断当前模型参数是否在预训练参数中
                    if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                        model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                        load_count += 1
                # 严格加载模型参数
                self.model.load_state_dict(model_state_dict, strict=True)
                # 输出权重参数加载率
                print("{:.2f}%的模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)))
                # 将预训练权重加载情况记录到日志文件
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("{:.2f}%的模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)), self.log_txt_path)
