# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2024/12/12 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import cv2
from tqdm import tqdm
import pandas as pd
import torch
from scipy import stats

from lib.utils import *
import lib.transforms.two as my_transforms
from lib import utils, dataloaders, models, metrics, testers


params_DRIVE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.16155026, 0.26819696, 0.50784565),
    "normalize_stds": (0.10571646, 0.18532471, 0.35080457),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "DRIVE",
    "dataset_path": r"./datasets/DRIVE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.08631576554733908, 0.913684234452661],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}

params_STARE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.11336552, 0.33381058, 0.58892505),
    "normalize_stds": (0.10905356, 0.19210595, 0.35295892),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "STARE",
    "dataset_path": r"./datasets/STARE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07542384887839432, 0.9245761511216056],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 2,
    "save_epoch_freq": 50,
}

params_CHASE_DB1 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.02789665, 0.16392259, 0.45287978),
    "normalize_stds": (0.03700363, 0.14539037, 0.36542216),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "CHASE-DB1",
    "dataset_path": r"./datasets/CHASE-DB1",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07186707540874207, 0.928132924591258],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}


def segment_image(params, model, image, label):
    transform = my_transforms.Compose(
        [
            my_transforms.Resize(params["resize_shape"]),
            my_transforms.ToTensor(),
            my_transforms.Normalize(
                mean=params["normalize_means"], std=params["normalize_stds"]
            ),
        ]
    )
    label[label == 255] = 1
    # 数据预处理和数据增强
    image, label = transform(image, label)
    # image扩充一维
    image = torch.unsqueeze(image, dim=0)
    # 转换数据格式
    label = label.to(dtype=torch.uint8)
    # 预测分割
    pred = torch.squeeze(model(image.to(params["device"])), dim=0)
    segmented_image_np = torch.argmax(pred, dim=0).to(dtype=torch.uint8).cpu().numpy()
    label_np = label.numpy()
    # image和numpy扩展到三维
    seg_image = np.dstack([segmented_image_np] * 3)
    label = np.dstack([label_np] * 3)
    # 定义红色、白色和绿色图像
    red = np.zeros((224, 224, 3))
    red[:, :, 0] = 255
    green = np.zeros((224, 224, 3))
    green[:, :, 1] = 255
    white = np.ones((224, 224, 3)) * 255
    segmented_display_image = np.zeros((224, 224, 3))
    segmented_display_image = np.where(
        seg_image & label, white, segmented_display_image
    )
    segmented_display_image = np.where(seg_image & ~label, red, segmented_display_image)
    segmented_display_image = np.where(
        ~seg_image & label, green, segmented_display_image
    )
    JI_score = cal_jaccard_index(seg_image, label)
    return segmented_display_image, JI_score


def generate_segment_result_images(
    dataset_name, model_names, seed=1777777, benchmark=False, deterministic=True
):
    # build the dir to save the result
    result_dir = os.path.join(r"./images", dataset_name + "_segment_result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    # select the dictionary of hyperparameters used for training
    if dataset_name == "DRIVE":
        params = params_DRIVE
    elif dataset_name == "STARE":
        params = params_STARE
    elif dataset_name == "CHASE-DB1":
        params = params_CHASE_DB1
    else:
        raise RuntimeError(f"No {dataset_name} dataset available")
    # launch initialization
    utils.reproducibility(seed, deterministic, benchmark)
    # define the dataset path
    dataset_root_dir = os.path.join(r"./datasets", dataset_name, "test")
    images_dir = os.path.join(dataset_root_dir, "images")
    labels_dir = os.path.join(dataset_root_dir, "annotations")
    cnt = 0
    # traverse each images
    for image_name in tqdm(os.listdir(images_dir)):
        filename, ext = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, filename + ".png")
        # 读取图像
        image = cv2.imread(image_path, -1)
        label = cv2.imread(label_path, -1)
        max_JI_score = 0
        max_model_name = None
        segment_result_images_list = []

        # traverse each models
        for model_name in model_names:
            params["model_name"] = model_name
            params["pretrain"] = os.path.join(
                r"./pretrain", dataset_name + "_" + model_name + ".pth"
            )
            # initialize model
            model = models.get_model(params)
            # load model weight
            pretrain_state_dict = torch.load(
                params["pretrain"],
                map_location=lambda storage, loc: storage.cuda(params["device"]),
            )
            model_state_dict = model.state_dict()
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (
                    model_state_dict[param_name].size()
                    == pretrain_state_dict[param_name].size()
                ):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
            model.load_state_dict(model_state_dict, strict=True)
            # segment
            seg_result_image, JI_score = segment_image(
                params, model, image.copy(), label.copy()
            )
            # save the segmented images
            segment_result_images_list.append(seg_result_image)
            # update max JI metric
            if JI_score > max_JI_score:
                max_JI_score = JI_score
                max_model_name = model_name
        # meet the conditions for preservation
        if max_model_name == "PMFSNet":
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                os.path.join(
                    result_dir,
                    "{:04d}".format(cnt) + "_0.jpg",
                ),
                image,
            )
            label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(
                    result_dir,
                    "{:04d}".format(cnt) + "_1.jpg",
                ),
                label,
            )
            for j, segment_result_image in enumerate(segment_result_images_list):
                segment_result_image = segment_result_image[:, :, ::-1]
                cv2.imwrite(
                    os.path.join(
                        result_dir,
                        "{:04d}".format(cnt) + "_" + str(j + 2) + ".jpg",
                    ),
                    segment_result_image,
                )
            cnt += 1


if __name__ == "__main__":
    # generate segmented images
    generate_segment_result_images(
        "DRIVE",
        ["UNet", "AttU_Net", "CANet", "CENet", "CPFNet", "CKDNet", "SwinUnet", "DATransUNet", "PMFSNet"]
    )
