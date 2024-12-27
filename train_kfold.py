# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2024/12/28 22:52
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import gc
import glob
import math
import tqdm
import shutil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from lib import utils, dataloaders, models, losses, metrics, trainers

params = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Kfold-3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "UNet3D",
    "in_channels": 1,
    "classes": 2,
    "scaling_version": "TINY",
    "dimension": "3d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["HD", "ASSD", "IoU", "SO", "DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "fold_k": 5,
    "start_fold": 0,
    "current_fold": 0,
    "metric_results_per_fold": {"HD": [], "ASSD": [], "IoU": [], "SO": [], "DSC": []},
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dsc": -1.0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}


def load_kfold_state():
    resume_state_dict = torch.load(params["resume"], map_location=lambda storage, loc: storage.cuda(params["device"]))
    params["start_fold"] = resume_state_dict["fold"]
    params["current_fold"] = resume_state_dict["fold"]
    params["metric_results_per_fold"] = resume_state_dict["metric_results_per_fold"]


def cross_validation(loss_function, metric):
    # initialize kfold evaluation
    kf = KFold(n_splits=params["fold_k"], shuffle=True, random_state=params["seed"])
    # get all paths of images and labels
    images_path_list = sorted(glob.glob(os.path.join(params["dataset_path"], "**", "images", "*.nii.gz")))
    labels_path_list = sorted(glob.glob(os.path.join(params["dataset_path"], "**", "labels", "*.nii.gz")))
    for i, (train_index, valid_index) in enumerate(kf.split(images_path_list, labels_path_list)):
        if i < params["start_fold"]:
            continue
        params["current_fold"] = i
        print("Start training {}-fold......".format(i))
        utils.pre_write_txt("Start training {}-fold......".format(i), params["log_txt_path"])

        # split dataset
        train_images_path_list, train_labels_path_list = list(np.array(images_path_list)[train_index]), list(np.array(labels_path_list)[train_index])
        valid_images_path_list, valid_labels_path_list = list(np.array(images_path_list)[valid_index]), list(np.array(labels_path_list)[valid_index])
        print([os.path.basename(path) for path in train_images_path_list], [os.path.basename(path) for path in valid_images_path_list])

        # initialize dataloader
        train_loader, valid_loader = dataloaders.get_dataloader(params, train_images_path_list, train_labels_path_list, valid_images_path_list, valid_labels_path_list)

        # initialize the model, optimizer, and lr_scheduler
        model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)

        # initialize the trainer
        trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

        # resume or load pretrained weights
        if (params["resume"] is not None) and (params["pretrain"] is not None) and (i == params["start_fold"]):
            trainer.load()

        # start training
        trainer.training()

        # gc memory
        gc.collect()


def calculate_metrics():
    result_str = "\n\n"
    for metric_name, values in params["metric_results_per_fold"].items():
        result_str += metric_name + ":"
        for value in values:
            result_str += "  " + str(value)
        result_str += "\n"
    utils.pre_write_txt(result_str, params["log_txt_path"])

    print_info = "\n\n"
    # metrics as columns
    print_info += " " * 12
    for metric_name in params["metric_names"]:
        print_info += "{:^12}".format(metric_name)
    print_info += '\n'
    # add the mean of metrics
    print_info += "{:<12}".format("mean:")
    for metric_name, values in params["metric_results_per_fold"].items():
        print_info += "{:^12.6f}".format(np.mean(np.array(values)))
    print_info += '\n'
    # add the std od metrics
    print_info += "{:<12}".format("std:")
    for metric_name, values in params["metric_results_per_fold"].items():
        print_info += "{:^12.6f}".format(np.std(np.array(values)))
    print(print_info)
    utils.pre_write_txt(print_info, params["log_txt_path"])


if __name__ == '__main__':
    # launch initialization
    # os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # get the cuda device
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("Complete the initialization of configuration")

    # initialize the loss function
    loss_function = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # whether load kfold state
    if (params["resume"] is not None) and (params["pretrain"] is not None):
        load_kfold_state()
        print("Complete load the kfold state")

    # ger path of run directory
    if params["resume"] is None:
        params["execute_dir"] = os.path.join(params["run_dir"],
                                             utils.datestr() +
                                             "_" + params["model_name"] +
                                             "_" + params["dataset_name"] +
                                             "_" + params["lr_scheduler_name"] + (str(params["patience"]) if params["lr_scheduler_name"] == "ReduceLROnPlateau" else ""))
    else:
        params["execute_dir"] = os.path.dirname(os.path.dirname(params["resume"]))
    params["checkpoint_dir"] = os.path.join(params["execute_dir"], "checkpoints")
    params["log_txt_path"] = os.path.join(params["execute_dir"], "log.txt")
    if params["resume"] is None:
        utils.make_dirs(params["checkpoint_dir"])

    # kfold training
    print("Start kfold training......")
    cross_validation(loss_function, metric)

    # Calculate the comprehensive results of the metrics
    calculate_metrics()
