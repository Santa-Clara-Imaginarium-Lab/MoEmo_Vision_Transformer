
# MoEmo Vision Transformer
MoEmo Vision Transformer is a new approach in HRI(human-robot interaction) because it uses cross-attention and movement vectors to enhance 3D pose estimation for __emotion detection__. Recent developments in HRI emphasize why robots need to understand human emotions. Most papers focus on facial expressions to recognize emotions, but we focus on human body movements, and consider context. Context is very important for emotions because the same pose with different contexts will show different emotions.

<div align="center">
    <img src="assest/datset.png", width="900">
</div>

# News!
- Nov 2023: Our paper's codes are released!
- Oct 2023: Our paper was accepted by IROS 2023 (IEEE/RSJ International Conference on Intelligent Robots and Systems).

# Installation
## Conda environment
```shell
conda create -n MoEmo python=3.7
conda activate MoEmo
```
## torch
[PyTorch >= 1.7](https://pytorch.org/) + [CUDA](https://developer.nvidia.com/cuda-downloads)
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## visuial

- FFmpeg (if you want to export MP4 videos)
- ImageMagick (if you want to export GIFs)

## 3D Pose Estimator

- tqdm
- pillow
- scipy
- pandas
- h5py
- visdom
- nibabel
- opencv-python (install with pip)
- matplotlib

# Pre-trained model
You need to download the 2D pose estimator for the P-STMO, and then download the Pre-trained P-STMO wild data model for the 3D pose estimator.

1. Git the 2D pose estimator codes
```
git clone https://github.com/zh-plus/video-to-pose3D.git
```

2. Download pre-trained ALphapose as a 2D pose estimator

- Download **duc_se.pth** from ([Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)),
         place to `./joints_detectors/Alphapose/models/sppe`


3. Download pre-trained YOLO as the human detection model
- In order to handle multi-person in videos, we apply YOLO in advance to detect humans in frames.

- Download **yolov3-spp.weights** from ([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)),
         place to `./joints_detectors/Alphapose/models/yolo`

4. Download the P-STMO codes
```
git clone https://github.com/paTRICK-swk/P-STMO.git
```

5. 
6.  
