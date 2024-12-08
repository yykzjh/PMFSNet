# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh
@Contact  :   1378453948@qq.com
@DateTime :   2024/12/08 17:05
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import pandas as pd
import torch
from scipy import stats

from lib import utils, dataloaders, models, metrics, testers

params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
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
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0.60,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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
    "batch_size": 1,
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
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1 - 0.029],
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


def run(params, dataset_ind, dataset_total, model_ind, model_total):
    # print current dataset and model
    print("[{:01d}/{:01d}] [{:02d}/{:02d}]Current dataset: {}, model_name: {}".format(dataset_ind, dataset_total, model_ind, model_total, params["dataset_name"], params["model_name"]))
    # initialize the dataloader
    valid_loader = dataloaders.get_test_dataloader(params)
    # initialize the model
    model = models.get_model(params)
    # initialize the metrics
    metric = metrics.get_metric(params)
    # initialize the tester
    tester = testers.get_tester(params, model, metric)
    # load training weights
    tester.load()
    # evaluate the dsc of valid set
    return tester.evaluate_all_metrics(valid_loader)


def main(seed=1777777, benchmark=False, deterministic=True):
    # launch initialization
    utils.reproducibility(seed, deterministic, benchmark)
    # define datasets
    datasets_list = ["3D-CBCT-Tooth", "ISIC-2018"]
    # define the corresponding models
    models_list = [
        ["UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"],
        ["UNet", "AttU_Net", "CANet", "BCDUNet", "CENet", "CPFNet", "CKDNet", "PMFSNet"]
    ]

    # traverse all datasets in turn
    for dataset_ind, dataset_name in enumerate(datasets_list):
        # define result dicts
        dsc_dict = {}
        iou_dict = {}
        # traverse all models in turn
        for model_ind, model_name in enumerate(models_list[dataset_ind]):
            # select the dictionary of hyperparameters used for training
            if dataset_name == "3D-CBCT-Tooth":
                params = params_3D_CBCT_Tooth
            elif dataset_name == "ISIC-2018":
                params = params_ISIC_2018
            else:
                raise RuntimeError(f"No {dataset_name} dataset available")
            # update the dictionary of hyperparameters used for training
            params["dataset_name"] = dataset_name
            params["dataset_path"] = os.path.join(r"./datasets", ("NC-release-data-checked" if dataset_name == "3D-CBCT-Tooth" else dataset_name))
            params["model_name"] = model_name
            params["pretrain"] = os.path.join(r"./pretrain", ("Tooth_" if dataset_name == "3D-CBCT-Tooth" else "ISIC2018_") + model_name + ".pth")
            # calculate the dsc and iou of all images in test set
            dsc_list, iou_list = run(params, dataset_ind + 1, len(datasets_list), model_ind + 1, len(models_list[dataset_ind]))
            dsc_dict[model_name] = dsc_list
            iou_dict[model_name] = iou_list
        # convert to dataframe
        dsc_df = pd.DataFrame.from_dict(dsc_dict)
        iou_df = pd.DataFrame.from_dict(iou_dict)
        # save the dsc dataframe
        dsc_df.to_excel(os.path.join(r"./files", dataset_name + "_dsc.xlsx"))
        iou_df.to_excel(os.path.join(r"./files", dataset_name + "_iou.xlsx"))


def calculate_p_value(dataset_name, metric_name, base_model_name):
    # load df
    file_path = os.path.join(r"./files", dataset_name + "_" + metric_name + ".xlsx")
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # extract base_model_name column
    base_model_scores = df[base_model_name]

    # define p_value dict
    p_values = {}

    # paired t-test for each of the other model columns
    for model_name in df.columns:
        # exclude base_model_name
        if model_name != base_model_name:
            model_scores = df[model_name]
            t_stat, p_value = stats.ttest_rel(model_scores, base_model_scores)
            p_values[model_name] = p_value

    # print p_value
    print("p-values relative to {}:".format(base_model_scores))
    for model_name, p_value in p_values.items():
        print(f"{model_name}: {p_value}")

    # save p_value
    p_value_df = pd.DataFrame(p_values)
    p_value_df.to_excel(os.path.join(r"./files", dataset_name + "_" + metric_name + "_pvalue" + ".xlsx"))

if __name__ == '__main__':
    # evaluate all metrics
    main(seed=1777777, benchmark=False, deterministic=True)

    # calculate p_value
    calculate_p_value(dataset_name="ISIC-2018", metric_name="dsc", base_model_name="PMFSNet")


