import os
import time
import numpy as np

import nni
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from lib import utils


class ToothTrainer:
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
        if self.opt["use_amp"]:
            # 初始化自动混合精度缩放器
            self.scaler = GradScaler()

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
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, purge_step=0, max_queue=1, flush_secs=30)

        # 训练时需要用到的参数
        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_dice = opt["best_dice"]
        self.update_weight_freq = opt["update_weight_freq"]
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

            # 更新学习率
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"])
            else:
                self.lr_scheduler.step()

            # epoch结束总的输出一下结果
            print("epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                  .format(epoch, self.end_epoch - 1,
                          self.optimizer.param_groups[0]['lr'],
                          self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                          self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                          self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"],
                          self.best_dice))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                                    .format(epoch, self.end_epoch-1,
                                            self.optimizer.param_groups[0]['lr'],
                                            self.statistics_dict["train"]["loss"]/self.statistics_dict["train"]["count"],
                                            self.statistics_dict["train"]["DSC"]["avg"]/self.statistics_dict["train"]["count"],
                                            self.statistics_dict["valid"]["DSC"]["avg"]/self.statistics_dict["valid"]["count"],
                                            self.best_dice), self.log_txt_path)
                self.write_statistcs(mode="epoch", iter=epoch)

            if self.opt["optimize_params"]:
                # 向nni上报每个epoch验证集的平均dsc作为中间指标
                nni.report_intermediate_result(
                    self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"])

        if self.opt["optimize_params"]:
            # 将在验证集上最优的dsc作为最终上报指标
            nni.report_final_result(self.best_dice)

        # 关闭tensorboard
        time.sleep(60)
        self.writer.close()



    def train_epoch(self, epoch):

        # 训练
        self.model.train()

        # 遍历数据集的batch
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):

            # 将输入图像和标注图像都移动到指定设备上
            input_tensor, target = input_tensor.to(self.device), target.to(self.device)

            if self.opt["use_amp"]:
                # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
                with autocast():

                    t0 = time.time()
                    # 前向传播
                    output = self.model(input_tensor)
                    t1 = time.time()
                    # 计算损失值
                    dice_loss = self.loss_function(output, target)
                    t2 = time.time()

                t3 = time.time()
                # 对loss进行缩放，针对缩放后的loss进行反向传播(此部分计算在autocast()作用范围以外)
                self.scaler.scale(dice_loss / self.update_weight_freq).backward()
                t4 = time.time()

                # 判断满不满足梯度累加的周期
                if (batch_idx + 1) % self.update_weight_freq == 0:
                    t5 = time.time()
                    # 将梯度值缩放回原尺度后，优化器更新参数
                    # self.scaler.step(self.optimizer)
                    self.optimizer.step()
                    t6 = time.time()
                    # 更新scalar的缩放信息
                    # self.scaler.update()
                    t7 = time.time()
                    # 梯度清0
                    self.optimizer.zero_grad()
            else:
                t0 = time.time()
                # 前向传播
                output = self.model(input_tensor)
                t1 = time.time()
                # 计算损失值
                dice_loss = self.loss_function(output, target)
                t2 = time.time()

                t3 = time.time()
                scaled_dice_loss = dice_loss / self.update_weight_freq
                scaled_dice_loss.backward()
                t4 = time.time()

                # 判断满不满足梯度累加的周期
                if (batch_idx + 1) % self.update_weight_freq == 0:
                    t5 = time.time()
                    # 将梯度值缩放回原尺度后，优化器更新参数
                    self.optimizer.step()
                    t6 = time.time()
                    # 梯度清0
                    self.optimizer.zero_grad()

            t8 = time.time()
            # 计算各评价指标并更新中间统计信息
            self.calculate_metric_and_update_statistcs(output.cpu().float(), target.cpu().float(), len(target), dice_loss.cpu(), mode="train")
            t9 = time.time()

            # 每一次参数更新都将统计数据写入到tensorboard
            if (batch_idx + 1) % self.update_weight_freq == 0 and (not self.opt["optimize_params"]):
                self.write_statistcs(mode="step", iter=epoch*len(self.train_data_loader)+batch_idx)

            # if (batch_idx + 1) % self.update_weight_freq == 0:
            #     print("----------------------------------batch 2----------------------------------")
            #     print("前向传播时间：{:.6f}秒".format(t1 - t0))
            #     print("计算loss时间：{:.6f}秒".format(t2 - t1))
            #     print("反向传播时间：{:.6f}秒".format(t4 - t3))
            #     print("优化器更新参数时间：{:.6f}秒".format(t6 - t5))
            #     if self.opt["use_amp"]:
            #         print("更新scaler时间：{:.6f}秒".format(t7 - t6))
            #     print("计算评价指标时间：{:.6f}秒".format(t9 - t8))
            # else:
            #     print("----------------------------------batch 1----------------------------------")
            #     print("前向传播时间：{:.6f}秒".format(t1 - t0))
            #     print("计算loss时间：{:.6f}秒".format(t2 - t1))
            #     print("反向传播时间：{:.6f}秒".format(t4 - t3))
            #     print("计算评价指标时间：{:.6f}秒".format(t9 - t8))
            # if batch_idx == 3:
            #     exit()

            # 判断满不满足打印信息或者画图表的周期
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                print("epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}"
                      .format(epoch, self.end_epoch-1,
                              batch_idx+1, len(self.train_data_loader),
                              self.optimizer.param_groups[0]['lr'],
                              self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"]))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}"
                                        .format(epoch, self.end_epoch-1,
                                                batch_idx+1, len(self.train_data_loader),
                                                self.optimizer.param_groups[0]['lr'],
                                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"]),
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
                output = self.split_forward(input_tensor, self.model)

                # 计算各评价指标并更新中间统计信息
                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target), mode="valid")

            # 计算当前epoch验证集的dsc
            cur_dsc = self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"]

            # 按照一定周期固定保存模型和训练状态部分
            if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0:
                self.save(epoch, cur_dsc, self.best_dice, type="normal")
            if not self.opt["optimize_params"]:
                # 每次都保存最新的latest
                self.save(epoch, cur_dsc, self.best_dice, type="latest")
            # 与最优结果进行比较，保存最优的模型
            if cur_dsc > self.best_dice:
                self.best_dice = cur_dsc
                if not self.opt["optimize_params"]:
                    self.save(epoch, cur_dsc, self.best_dice, type="best")


    def write_statistcs(self, mode="step", iter=None):
        """
        将统计信息写入到tensorboard图表中

        :param mode: "step"训练时每一步的数据，"epoch"每个epoch结束后的数据
        :param iter: 当前step次数
        :return:
        """
        if mode == "step":
            # 写入dice_loss
            self.writer.add_scalar("step_train_loss",
                                   self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                   iter)
            # 写入dsc评价指标
            self.writer.add_scalar("step_train_dsc",
                                   self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                                   iter)
        else:
            # 写入epoch_loss
            self.writer.add_scalar("epoch_train_loss",
                                   self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                   iter)
            # 写入train_dsc
            self.writer.add_scalar("epoch_train_dsc",
                                   self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                                   iter)
            # 写入valid_dsc
            self.writer.add_scalar("epoch_valid_dsc",
                                   self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"],
                                   iter)


    def split_forward(self, image, model):
        """
        对于验证集完整图像，需要滑动切块后分别进行预测，最后再拼接到一起

        Args:
            image: 验证集完整图像
            model: 网络模型

        Returns:

        """
        # 获取图像尺寸
        ori_shape = image.size()[2:]
        # 初始化输出的特征图
        output = torch.zeros((image.size()[0], self.opt["classes"], *ori_shape), device=image.device)
        # 切片的大小
        slice_shape = self.opt["crop_size"]
        # 在三个维度上滑动的步长
        stride = self.opt["crop_stride"]

        # 在三个维度上进行滑动切片
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
        for i, metric_name in enumerate(self.opt["metric_names"]):
            # 计算当前评价指标在各类别上的数值
            per_class_metric = self.metric[i](output, target)
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
            self.best_dice = resume_state_dict["best_metric"]
            # 加载优化器参数
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            # 加载学习率调度器参数
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            # 加载模型参数字典
            model_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            # 严格加载模型参数
            self.model.load_state_dict(model_state_dict, strict=True)
        else:  # 如果不需要继续训练
            # 有可能需要加载模型的预训练参数
            if self.opt["pretrain"] is not None:
                # 加载模型参数字典
                model_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                # 严格加载模型参数
                self.model.load_state_dict(model_state_dict, strict=True)



