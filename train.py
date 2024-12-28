# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/12/30 17:05
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import argparse
import os

import nni
import torch

from lib import utils, dataloaders, models, losses, metrics, trainers

params_3D_CBCT_Tooth = {
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
    "samples_train": 1024,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Compose",
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
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 4,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
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
    "patience": 50,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": True,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 3,
    "best_dice": 0.60,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_MMOTU = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.12097393901893663,
    "color_jitter": 0.4203933474361258,
    "random_rotation_angle": 30,
    "normalize_means": (0.22250386, 0.21844882, 0.21521868),
    "normalize_stds": (0.21923075, 0.21622984, 0.21370508),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "MMOTU",
    "dataset_path": r"./datasets/MMOTU",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "momentum": 0.7725414416309884,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8689275449032848,
    "step_size": 5,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 5,
    "mode": "max",
    "patience": 1,
    "factor": 0.97,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.2350689696563569, 1-0.2350689696563569],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 2000,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 500,
}

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "ISIC-2018",
    "dataset_path": r"./datasets/ISIC-2018",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 100,
    "T_0": 5,
    "T_mult": 5,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1-0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
}

params_DRIVE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
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
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
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
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
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
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
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

params_Kvasir_SEG = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.22448543324157222,
    "color_jitter": 0.3281010563062837,
    "random_rotation_angle": 30,
    "normalize_means": (0.24398195, 0.32772844, 0.56273),
    "normalize_stds": (0.18945072, 0.2217485, 0.31491405),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Kvasir-SEG",
    "dataset_path": r"./datasets/Kvasir-SEG",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.000001,
    "momentum": 0.7781834740942233,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8079569870480704,
    "step_size": 20,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 500,
    "T_0": 10,
    "T_mult": 2,
    "mode": "max",
    "patience": 5,
    "factor": 0.91,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.1557906849111095, 0.8442093150888904],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 500,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 150,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="3D-CBCT-Tooth", help="dataset name")
    parser.add_argument("-m", "--model", type=str, default="PMFSNet", help="model name")
    parser.add_argument("-pre", "--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("-dim", "--dimension", type=str, default="2d", help="dimension of dataset images and models")
    parser.add_argument("-s", "--scaling_version", type=str, default="TINY", help="scaling version of PMFSNet")
    parser.add_argument("--epoch", type=int, default=None, help="training epoch")
    args = parser.parse_args()
    return args



def main():
    # analyse console arguments
    args = parse_args()

    # select the dictionary of hyperparameters used for training
    if args.dataset == "3D-CBCT-Tooth":
        params = params_3D_CBCT_Tooth
    elif args.dataset == "MMOTU":
        params = params_MMOTU
    elif args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    elif args.dataset == "DRIVE":
        params = params_DRIVE
    elif args.dataset == "STARE":
        params = params_STARE
    elif args.dataset == "CHASE-DB1":
        params = params_CHASE_DB1
    elif args.dataset == "Kvasir-SEG":
        params = params_Kvasir_SEG
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", args.dataset)
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = args.epoch // 4


    if params["optimize_params"]:
        # 获得下一组搜索空间中的参数
        tuner_params = nni.get_next_parameter()
        # 更新参数
        params.update(tuner_params)

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

    # initialize the dataloader
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("Complete the initialization of dataloader")

    # initialize the model, optimizer, and lr_scheduler
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # initialize the loss function
    loss_function = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # initialize the trainer
    trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

    # resume or load pretrained weights
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()
    print("Complete the initialization of trainer")

    # start training
    trainer.training()



if __name__ == '__main__':
    main()


