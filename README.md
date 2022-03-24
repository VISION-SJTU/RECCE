# RECCE CVPR 2022

:page_facing_up: End-to-End Reconstruction-Classification Learning for Face Forgery Detection

:boy: Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, Xiaokang Yang

**Please consider citing our paper if you find it interesting or helpful to your research.**
```
Bibtex will come up soon ...
```

----

![RECCE Framework](figure/framework.png)

### Introduction

This repository is an implementation for *End-to-End Reconstruction-Classification Learning for Face Forgery Detection* presented in CVPR 2022. In the paper, we propose a novel **REC**onstruction-**C**lassification l**E**arning framework called **RECCE** to detect face forgeries. The code is based on Pytorch. Please follow the instructions below to get started.


### Basic Requirements
Please ensure that you have already installed the following packages.
- [Pytorch](https://pytorch.org/get-started/previous-versions/) 1.7.1
- [Torchvision](https://pytorch.org/get-started/previous-versions/) 0.8.2
- [Albumentations](https://github.com/albumentations-team/albumentations#spatial-level-transforms) 1.0.3
- [Timm](https://github.com/rwightman/pytorch-image-models) 0.3.4
- [TensorboardX](https://pypi.org/project/tensorboardX/#history) 2.1
- [Scipy](https://pypi.org/project/scipy/#history) 1.5.2
- [PyYaml](https://pypi.org/project/PyYAML/#history) 5.3.1

### Dataset Preparation
- We include the dataset loaders for several commonly-used face forgery datasets, *i.e.,* [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html), [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild), and [DFDC](https://ai.facebook.com/datasets/dfdc). You can enter the dataset website to download the original data.
- For FaceForensics++, Celeb-DF, and DFDC, since the original data are in video format, you should first extract the facial images from the sequences and store them. We use [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) to do this.

### Config Files
- We have already provided the config templates in `config/`. You can adjust the parameters in the yaml files to specify a training process. More information is presented in [config/README.md](./config/README.md).

### Training
- We use `torch.distributed` package to train the models, for more information, please refer to [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html).
- To train a model, run the following script in your console. 
```{bash}
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 train.py --config path/to/config.yaml
```
- `--config`: Specify the path of the config file. 

### Testing
- To test a model, run the following script in your console. 
```{bash}
python test.py --config path/to/config.yaml
```
- `--config`: Specify the path of the config file.

### Acknowledgement
- We thank Qiqi Gu for helping plot the schematic diagram of the proposed method in the manuscript.