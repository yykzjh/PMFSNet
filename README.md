# PMFSNet

PMFSNet : Polarized Multi-scale Feature Self-attention Network For Lightweight Medical Image Segmentation
![PMFSNet.png](https://github.com/yykzjh/PMFSNet/blob/master/images/PMFSNet.png)
![3D_CBCT_Tooth_bubble_image.jpg](https://github.com/yykzjh/PMFSNet/blob/master/images/3D_CBCT_Tooth_bubble_image.jpg)

## To-do List
- [x] Training and testing code
- [x] Training weights of other models on our experiments 
- [x] Pre-training weights of PMFSNet on ImageNet2012

## Environment
Please prepare an environment with python=3.8:
```
conda create -n py38 python=3.8
conda activate py38
git clone git@github.com:yykzjh/PMFSNet.git
cd PMFSNet
```
Please install the environment dependencies by `requirements.txt`:
```
pip install -r requirements.txt
```

## Data Preparation
We obtained three public datasets and processed them in some way. All datasets are placed in the `./datasets` directory after unzipping.
1. [3D CBCT Tooth dataset](https://pan.baidu.com/s/10qf6k10GE9OHYcJ76wrx-w?pwd=6ad8):
```
./datasets/3D-CBCT-Tooth/
	sub_volumes/160-160-96_2048.npz
	train/
		images/
			1000889125_20171009.nii.gz
			......
			X2360674.nii.gz
		labels/
			1000889125_20171009.nii.gz
			......
			X2360674.nii.gz
	valid/
		images/
			1000813648_20180116.nii.gz
			......
			X2358714.nii.gz
		labels/
			1000813648_20180116.nii.gz
			......
			X2358714.nii.gz
```
2. [MMOTU](https://pan.baidu.com/s/10AT7fqgbK2s507tr1MfpTQ?pwd=mo3c):
```
./datasets/MMOTU/
	train/
		images/
			1.JPG
			......
			1465.JPG
		labels/
			1.PNG
			......
			1465.PNG
	valid/
		images/
			3.JPG
			......
			1469.JPG
		labels/
			3.PNG
			......
			1469.PNG
```
3. [ISIC 2018](https://pan.baidu.com/s/16vla-i12GSwjqTTGc0CXSA?pwd=qola):
```
./datasets/ISIC-2018/
	train/
		images/
			ISIC_0000000.jpg
			......
			ISIC_0016072.jpg
		annotations/
			ISIC_0000000_segmentation.png
			......
			ISIC_0016072_segmentation.png
	test/
		images/
			ISIC_0000003.jpg
			......
			ISIC_0016060.jpg
		annotations/
			ISIC_0000003_segmentation.png
			......
			ISIC_0016060_segmentation.png
```

## Training
Running `train.py` script can easily start the training.  Customize the training by passing in the following arguments:
```
--dataset: dataset name, optional 3D-CBCT-Tooth, MMOTU, ISIC-2018
--model: model name, see below implemented architectures for details
--pretrain_weight: pre-trained weight file path
--dimension: dimension of dataset images and models, for PMFSNet only
--scaling_version: scaling version of PMFSNet, for PMFSNet only
--epoch: training epoch
```
Training demo:
```python
python ./train.py --dataset 3D-CBCT-Tooth --model PMFSNet --dimension 3d --scaling_version TINY --epoch 20
python ./train.py --dataset MMOTU --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-basic_ILSVRC2012.pth --dimension 2d --scaling_version BASIC --epoch 2000
python ./train.py --dataset ISIC-2018 --model PMFSNet --dimension 2d --scaling_version BASIC --epoch 150
```

## Testing
### Evaluating model performance on a test set
Running `test.py` script to start the evaluation. The input arguments are the same as the training script. Testing demo:
```python
python ./test.py --dataset 3D-CBCT-Tooth --model PMFSNet --pretrain_weight ./pretrain/PMFSNet3D-TINY_Tooth.pth --dimension 3d --scaling_version TINY
python ./test.py --dataset MMOTU --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-BASIC_MMOTU.pth --dimension 2d --scaling_version BASIC
python ./test.py --dataset ISIC-2018 --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-BASIC_ISIC2018.pth --dimension 2d --scaling_version BASIC
```
### Inferring a single image segmentation result
Running `inference.py` script to start the inference. The additional argument `--image_path` denotes the path of the inferred image. Inferring demo:
```python
python ./inference.py --dataset 3D-CBCT-Tooth --model PMFSNet --pretrain_weight ./pretrain/PMFSNet3D-TINY_Tooth.pth --dimension 3d --scaling_version TINY --image_path ./images/1001250407_20190923.nii.gz
python ./inference.py --dataset MMOTU --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-BASIC_MMOTU.pth --dimension 2d --scaling_version BASIC --image_path ./images/453.JPG
python ./inference.py --dataset ISIC-2018 --model PMFSNet --pretrain_weight ./pretrain/PMFSNet2D-BASIC_ISIC2018.pth --dimension 2d --scaling_version BASIC --image_path ./images/ISIC_0000550.jpg
```

## Implemented Architectures
+ 3D CBCT Tooth:
	- [UNet3D](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49): 3D U-Net: learning dense volumetric segmentation from sparse annotation
	- [VNet](https://ieeexplore.ieee.org/abstract/document/7785132): V-net: Fully convolutional neural networks for volumetric medical image segmentation
	- [DenseVNet](https://ieeexplore.ieee.org/abstract/document/8291609): Automatic multi-organ segmentation on abdominal CT with dense V-networks
	- [AttentionUNet3D](https://arxiv.org/pdf/1804.03999.pdf): Attention u-net: Learning where to look for the pancreas
	- [DenseVoxelNet](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_33): Automatic 3D cardiovascular MR segmentation with densely-connected volumetric convnets
	- [MultiResUNet3D](https://www.sciencedirect.com/science/article/abs/pii/S0893608019302503): MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation
	- [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf): Unetr: Transformers for 3d medical image segmentation
	- [SwinUNETR](https://link.springer.com/chapter/10.1007/978-3-031-08999-2_22): Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images
	- [TransBTS](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_11): Transbts: Multimodal brain tumor segmentation using transformer
	- [nnFormer](https://ieeexplore.ieee.org/abstract/document/10183842): nnFormer: volumetric medical image segmentation via a 3D transformer
	- [3DUXNet](https://arxiv.org/pdf/2209.15076.pdf): 3d ux-net: A large kernel volumetric convnet modernizing hierarchical transformer for medical image segmentation
+ MMOTU
	- [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf): Mobilenetv2: Inverted residuals and linear bottlenecks
	- [PSPNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf): Pyramid scene parsing network
	- [DANet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf): Dual attention network for scene segmentation
	- [SegFormer](https://proceedings.neurips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf): SegFormer: Simple and efficient design for semantic segmentation with transformers
	- [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28): U-net: Convolutional networks for biomedical image segmentation
	- [TransUNet](https://arxiv.org/pdf/2102.04306.pdf): Transunet: Transformers make strong encoders for medical image segmentation
	- [BiSeNetV2](https://link.springer.com/article/10.1007/s11263-021-01515-2): Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation
	- [MedT](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_4): Medical transformer: Gated axial-attention for medical image segmentation
+ ISIC 2018
	- [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28): U-net: Convolutional networks for biomedical image segmentation
	- [AttU_Net](https://arxiv.org/pdf/1804.03999.pdf): Attention u-net: Learning where to look for the pancreas
	- [CANet](https://ieeexplore.ieee.org/abstract/document/9246575): CA-Net: Comprehensive attention convolutional neural networks for explainable medical image segmentation
	- [BCDUNet](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Azad_Bi-Directional_ConvLSTM_U-Net_with_Densley_Connected_Convolutions_ICCVW_2019_paper.pdf): Bi-directional ConvLSTM U-Net with densley connected convolutions
	- [CENet](https://ieeexplore.ieee.org/abstract/document/8662594): Ce-net: Context encoder network for 2d medical image segmentation
	- [CPFNet](https://ieeexplore.ieee.org/abstract/document/9049412): CPFNet: Context pyramid fusion network for medical image segmentation
	- [CKDNet](https://www.sciencedirect.com/science/article/abs/pii/S156849462030819X): Cascade knowledge diffusion network for skin lesion diagnosis and segmentation

## Results
### 3D CBCT Tooth
Comparison results of different methods on the 3D CBCT tooth dataset.
| Method               | FLOPs(G) | Params(M) | HD(mm) | ASSD(mm) | IoU(%) | SO(%) | DSC(%) |          Weights          |
| -------------------- | :------: | :-------: | :----: | :------: | :----: | :---: | :----: | :-----------------------: |
| UNet3D               | 2223.03  |   16.32   | 113.79 |  22.40   | 70.62  | 70.72 | 36.67  |     [UNet3D_Tooth](https://pan.baidu.com/s/1TuR6KFWkov35P2tU9hDn0w?pwd=28q2)      |
| DenseVNet            |  23.73   |   0.87    |  8.21  |   1.14   | 84.57  | 94.88 | 91.15  |    [DenseVNet_Tooth](https://pan.baidu.com/s/15AoxmLgyIS2T7ubrKA8zBQ?pwd=ixog)    |
| AttentionUNet3D      | 2720.79  |   94.48   | 147.10 |  61.10   | 52.52  | 42.49 | 64.08  | [AttentionUNet3D_Tooth](https://pan.baidu.com/s/1Ga2ONiGIvVHSa_ZnXjjUNg?pwd=d90h) |
| DenseVoxelNet        |  402.32  |   1.78    | 41.18  |   3.88   | 81.51  | 92.50 | 89.58  |  [DenseVoxelNet_Tooth](https://pan.baidu.com/s/1oPhbRUrqRY5oHtjsOVis0g?pwd=d99o)  |
| MultiResUNet3D       | 1505.38  |   17.93   | 74.06  |   8.17   | 76.19  | 81.70 | 65.45  | [MultiResUNet3D_Tooth](https://pan.baidu.com/s/1xI3IizurhcEb-8zrDdrm2g?pwd=1da7)  |
| UNETR                |  229.19  |   93.08   | 107.89 |  17.95   | 74.30  | 73.14 | 81.84  |      [UNETR_Tooth](https://pan.baidu.com/s/1Kj3gSKl0u0SjCfOTEP508g?pwd=nerh)      |
| SwinUNETR            |  912.35  |   62.19   | 82.71  |   7.50   | 83.10  | 86.80 | 89.74  |    [SwinUNETR_Tooth](https://pan.baidu.com/s/18K0l2Pt3RzbpkiaKqV44Bg?pwd=pa6m)    |
| TransBTS             |  306.80  |   33.15   | 29.03  |   4.10   | 82.94  | 90.68 | 39.32  |    [TransBTS_Tooth](https://pan.baidu.com/s/1dxtb7w0J2W690SABfUrgZg?pwd=h9oh)     |
| nnFormer             |  583.49  |  149.25   | 51.28  |   5.08   | 83.54  | 90.89 | 90.66  |    [nnFormer_Tooth](https://pan.baidu.com/s/1mIEyQyE1rkvGwGuYLAZq1A?pwd=omxl)     |
| 3D UX-Net            | 1754.79  |   53.01   | 108.52 |  19.69   | 75.40  | 73.48 | 84.89  |     [3DUXNet_Tooth](https://pan.baidu.com/s/1DUGdIC6HYj47cpK-UhTk-g?pwd=737v)     |
| PMFSNet3D-TINY(Ours) |  15.14   |   0.63    |  5.57  |   0.79   | 84.68  | 95.10 | 91.30  |                           |

![3D_CBCT_Tooth_segmentation.jpg](https://github.com/yykzjh/PMFSNet/blob/master/images/3D_CBCT_Tooth_segmentation.jpg)

### MMOTU
Pre-training weights of PMFSNet2D-BASIC, PMFSNet2D-SMALL, and PMFSNet2D-TINY on ImageNet2012 dataset.
| Method          |            Weights             |
| --------------- | :----------------------------: |
| PMFSNet2D-BASIC | [PMFSNet2D-BASIC_ILSVRC2012](https://pan.baidu.com/s/101_wth3SVurWdkVk5aWXYA?pwd=ffl7) |
| PMFSNet2D-SMALL | [PMFSNet2D-SMALL_ILSVRC2012](https://pan.baidu.com/s/1iL57OXzP4utd5G-xz9eoow?pwd=z0ql) |
| PMFSNet2D-TINY  | [PMFSNet2D-TINY_ILSVRC2012](https://pan.baidu.com/s/1BwgyVDsDoECsIrLFoAOoHQ?pwd=t24c)  |

Comparison results of different methods on the MMOTU dataset.
| Method                | FLOPs(G) | Params(M) | IoU(%) | mIoU(%) | Weights |
| --------------------- | :------: | :-------: | :----: | :-----: | :-----: |
| PSPNet                |  38.71   |   53.32   | 82.01  |  89.41  |         |
| DANet                 |  10.95   |   47.44   | 82.20  |  89.53  |         |
| SegFormer             |   2.52   |   7.72    | 82.46  |  89.88  |         |
| U-Net                 |  41.90   |   31.04   | 79.91  |  86.80  |         |
| TransUNet             |  24.61   |  105.28   | 81.31  |  89.01  |         |
| BiseNetV2             |   3.40   |   5.19    | 79.37  |  86.13  |         |
| PMFSNet2D-BASIC(Ours) |   2.21   |   0.99    | 82.02  |  89.36  |         |

![MMOTU_segmentation.jpg](https://github.com/yykzjh/PMFSNet/blob/master/images/MMOTU_segmentation.jpg)

### ISIC 2018
Comparison results of different methods on the ISIC 2018 dataset.

| Method                | FLOPs(G) | Params(M) | IoU(%) | DSC(%) | ACC(%) | Weights               |
| --------------------- | :------: | :-------: | :----: | :----: | :----: | --------------------- |
| U-Net                 |  41.93   |   31.04   | 76.77  | 86.55  | 95.00  | [UNet_ISIC2018](https://pan.baidu.com/s/1Xg7JHIHGog4wC4WWlTCpMw?pwd=gdol)     |
| AttU-Net              |  51.07   |   34.88   | 78.19  | 87.54  | 95.33  | [AttU_Net_ISIC2018](https://pan.baidu.com/s/1Mt_P-rU78bGKXyYQj7cfrg?pwd=sg7d) |
| CA-Net                |   4.62   |   2.79    | 68.82  | 80.96  | 92.96  | [CANet_ISIC2018](https://pan.baidu.com/s/1z_W16tIerC5SQNK8BhsQ4g?pwd=fbwa)    |
| BCDU-Net              |  31.96   |   18.45   | 76.46  | 86.26  | 95.19  | [BCDUNet_ISIC2018](https://pan.baidu.com/s/1oXLdbbnp3n5L0I_RtdakEQ?pwd=9jrz)  |
| CE-Net                |   6.83   |   29.00   | 78.05  | 87.47  | 95.40  | [CENet_ISIC2018](https://pan.baidu.com/s/10HnUdmAro9WA9AUjpEbyDg?pwd=6s6d)    |
| CPF-Net               |   6.18   |   43.27   | 78.47  | 87.70  | 95.52  | [CPFNet_ISIC2018](https://pan.baidu.com/s/1EBSsLuKxIm_x3c5X_3GLpA?pwd=rk7t)   |
| CKDNet                |  12.69   |   59.34   | 77.89  | 87.35  | 95.27  | [CKDNet_ISIC2018](https://pan.baidu.com/s/1T2Mu0dLaYpD66p6q9G4UDQ?pwd=9zzf)   |
| PMFSNet2D-BASIC(Ours) |   2.21   |   0.99    | 78.82  | 87.92  | 95.59  |                       |

![ISIC2018_segmentation.jpg](https://github.com/yykzjh/PMFSNet/blob/master/images/ISIC2018_segmentation.jpg)

## References
+ [MONAI](https://github.com/Project-MONAI/MONAI/tree/dev)
+ [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
+ [Pytorch-Segmentation-multi-models](https://github.com/Minerva-J/Pytorch-Segmentation-multi-models)
+ [MedicalSeg](https://github.com/920232796/MedicalSeg)
+ [research-contributions](https://github.com/Project-MONAI/research-contributions)
+ [Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)
+ [ISIC 2018](https://challenge.isic-archive.com/data/#2018)
+ [surface-distance](https://github.com/google-deepmind/surface-distance)