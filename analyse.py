# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2024/12/13 23:40
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from proplot import rc
import matplotlib.pyplot as plt

import torch
from thop import profile
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary

import lib.models as models
import lib.dataloaders as dataloaders


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
    "metric_names": ["DSC"],
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
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}


def generate_segmented_sample_image(scale=1):
    image = np.full((976, 4010, 3), 255)
    for i in range(4):
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for j in range(12):
            img = cv2.imread(r"./images/NC-release-data_segment_result_samples/" + str(i) + "_{:02d}".format(j + 1) + ".jpg")
            img = np.rot90(img, -1)
            if j == 0:
                h, w, _ = img.shape
                y_points, x_points, _ = np.nonzero(img)
                x_min, x_max, y_min, y_max = x_points.min(), x_points.max(), y_points.min(), y_points.max()
            img = img[max(0, y_min - 1):min(h - 1, y_max + 2), max(0, x_min - 1):min(w - 1, x_max + 2), :]
            img = cv2.resize(img, (320, 224))
            pos_x, pos_y = i * (224 + 10), j * (320 + 10) + 60
            image[pos_x: pos_x + 224, pos_y: pos_y + 320, :] = img
    image = image[:, :, ::-1]

    col_names = ["Ground Truth", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"]
    row_names = ["(a)", "(b)", "(c)", "(d)"]
    col_positions = [115, 490, 795, 1080, 1425, 1745, 2140, 2435, 2780, 3125, 3450, 3780]
    row_positions = [100, 334, 568, 802]

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    for i, text in enumerate(col_names):
        position = (col_positions[i], 931)
        draw.text(position, text, font=font, fill=color)
    for i, text in enumerate(row_names):
        position = (5, row_positions[i])
        draw.text(position, text, font=font, fill=color, stroke_width=1)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/NC-release-data_segment_result_samples/3D_CBCT_Tooth_segmentation.jpg")



def generate_bubble_image():

    def circle_area_func(x, p=75, k=150):
        return np.where(x < p, (np.sqrt(x / p) * p) * k, x * k)

    def inverse_circle_area_func(x, p=75, k=150):
        return np.where(x < p * k, (((x / k) / p) ** 2) * p, x / k)


    rc["font.family"] = "Times New Roman"
    rc["axes.labelsize"] = 36
    rc["tick.labelsize"] = 32
    rc["suptitle.size"] = 28
    rc["title.size"] = 28

    data = pd.read_excel(r"./files/experience_data.xlsx", sheet_name="data01")
    model_names = data.Method
    FLOPs = data.FLOPs
    Params = data.Params
    values = data.IoU
    xtext_positions = [2250, 20, 2300, 1050, 225, 1010, 200, 600, 1810, 360, 10]
    ytext_positions = [68, 80.5, 54, 73, 70, 82, 83.5, 84, 74, 78, 86]
    legend_sizes = [1, 5, 25, 50, 100, 150]
    legend_yposition = 57.5
    legend_xpositions = [590, 820, 1030, 1260, 1480, 1710]
    p = 15
    k = 150

    fig, ax = plt.subplots(figsize=(18, 9), dpi=100, facecolor="w")
    pubble = ax.scatter(x=FLOPs, y=values, s=circle_area_func(Params, p=p, k=k), c=list(range(len(model_names))), cmap=plt.cm.get_cmap("Spectral"), lw=3, ec="white", vmin=0, vmax=11)
    center = ax.scatter(x=FLOPs[:-1], y=values[:-1], s=20, c="#e6e6e6")
    ours_ = ax.scatter(x=FLOPs[-1:], y=values[-1:], s=60, marker="*", c="red")

    for i in range(len(FLOPs)):
        ax.annotate(model_names[i], xy=(FLOPs[i], values[i]), xytext=(xtext_positions[i], ytext_positions[i]), fontsize=32, fontweight=(200 if i < (len(FLOPs)-1) else 600))
    for i, legend_size in enumerate(legend_sizes):
        ax.text(legend_xpositions[i], legend_yposition, str(legend_size) + "M", fontsize=32, fontweight=200)

    kw = dict(prop="sizes", num=legend_sizes, color="#e6e6e6", fmt="{x:.0f}", linewidth=None, markeredgewidth=3, markeredgecolor="white", func=lambda s: np.ceil(inverse_circle_area_func(s, p=p, k=k)))
    legend = ax.legend(*pubble.legend_elements(**kw), bbox_to_anchor=(0.7, 0.15), title="Parameters (Params) / M", ncol=6, fontsize=0, title_fontsize=0, handletextpad=90, frameon=False)

    ax.set(xlim=(0, 2900), ylim=(45, 90), xticks=np.arange(0, 2900, step=300), yticks=np.arange(45, 90, step=5), xlabel="Floating-point Operations Per Second (FLOPs) / GFLOPs", ylabel="Intersection over Union (IoU) / %")

    fig.tight_layout()
    fig.savefig("./images/3D_CBCT_Tooth_bubble_image.jpg", bbox_inches='tight', dpi=300)
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyse_models(model_names_list):
    opt = {
        "dataset_name": "DRIVE",
        "in_channels": 3,
        "classes": 2,
        "resize_shape": (224, 224),
        "scaling_version": "BASIC",
        "dimension": "2d",
        "device": "cuda:0",
    }
    for model_name in model_names_list:
        opt["model_name"] = model_name
        model = models.get_model(opt)

        print("***************************************** model name: {} *****************************************".format(model_name))

        print("params: {:.6f} M".format(count_parameters(model)/1e6))

        input = torch.randn(1, 3, opt["resize_shape"][0], opt["resize_shape"][1]).to(opt["device"])
        flops, params = profile(model, (input,))
        print("flops: {:.6f} G, params: {:.6f} M".format(flops / 1e9, params / 1e6))

        flops, params = get_model_complexity_info(model, (3, opt["resize_shape"][0], opt["resize_shape"][1]), as_strings=False, print_per_layer_stat=False)
        print("flops: {:.6f} G, params: {:.6f} M".format(flops / 1e9, params / 1e6))

        print(summary(model, input, show_input=False, show_hierarchical=False))


def analyse_valid_set():
    valid_loader = dataloaders.get_test_dataloader(params_3D_CBCT_Tooth)
    for batch_idx, (input_tensor, target) in enumerate(valid_loader):
        unique_index = torch.unique(target)
        print(unique_index)


def analyse_train_set():
    train_loader, valid_loader = dataloaders.get_dataloader(params_3D_CBCT_Tooth)
    for batch_idx, (input_tensor, target) in enumerate(train_loader):
        unique_index = torch.unique(target)
        print(unique_index)
        for index in unique_index:
            print(index.item())



if __name__ == '__main__':
    # generate_segmented_sample_image()

    # generate_bubble_image()

    # analyse_models(["UNet", "AttU_Net", "CANet", "CENet", "CPFNet", "CKDNet", "SwinUnet", "DATransUNet", "PMFSNet"])

    # analyse_valid_set()
    analyse_train_set()