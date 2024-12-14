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


if __name__ == '__main__':
    generate_segmented_sample_image()

    # generate_bubble_image()

    # analyse_models(["UNet", "AttU_Net", "CANet", "CENet", "CPFNet", "CKDNet", "SwinUnet", "DATransUNet", "PMFSNet"])