# CurrI2P

## Get Started

### Installation and Data Preparation

#### step 1. Please prepare environment

We implement our method on two baselines, and their environments are the same as their baselines. Therefore, you can refer to:

- [VP2P](https://github.com/junshengzhou/VP2P-Match)
- [CorrI2P](https://github.com/rsy6318/CorrI2P)

The inference code was tested on:

- Ubuntu 16.04 LTS, Python 3.7, Pytorch 1.7.1, CUDA 9.2, GeForce RTX 2080Ti (pip, Conda)

```
conda create -n CurrI2P python=3.7
conda activate CurrI2P
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

#### step 2. Prepare CurrI2P repo by.

```
git clone https://github.com/lin-liwei/CurrI2P.git
cd CurrI2P
```

#### step 3. Download data. 

We provide the inference data from the KITTI dataset (sequences 9-10) to help you quickly get started with evaluating CurrI2P (VP2P). The data tree should be arranged as:

```
KITTI/
├── calib
    └── 09
    └── 10
├── data_odometry_color
    └── sequences 
        ├── 09
        ├── 10
├── data_odometry_velodyne
    └── sequences 
        ├── 09
        ├── 10
```



## Inference

- **Checkpoints.** We provide the pre-trained checkpoint of CurrI2P(VP2P) in [here](https://drive.google.com/file/d/1o-aw-iosdP8Cp01-PZuP--1VT_O3By_N/view?usp=drive_link).
- **Scripts.** After prepraring the code and checkpoints, you can simply evaluate CurrI2P(VP2P) by runing:

```python
python main.py
```

## Train

To train CurrI2P, begin by preparing the training data as outlined in [VP2P](https://github.com/junshengzhou/VP2P-Match) or [CorrI2P](https://github.com/rsy6318/CorrI2P). Additional instructions for training CurrI2P will be released soon.

## Acknowledgements

This implementation is based on / inspired by:

- [VP2P](https://github.com/junshengzhou/VP2P-Match)
- [CorrI2P](https://github.com/rsy6318/CorrI2P)
- [DeepI2P](https://github.com/lijx10/DeepI2P)
