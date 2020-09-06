## Description
This repository contains scripts to train (fine-tune) individual (local and remote), Siminet and EarlyDiscard models on PKU-MMD dataset. This code is cloned from the original repository of [3D-ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch). Please refer to the original repository for customization options.

## Requirements
The code is tested with the following software packages.

1. Python 3.6.3
2. Pytorch 1.4.0
3. Torchvision 0.2.0
4. Pillow 6.0.0
5. Sklearn 0.23.0
6. Cuda 10.0, cuDNN 7.6.4

## Getting started
1. Download pre-trained models on Kinetics

The pre-trained 3D-ResNet models on Kinetics are downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing). Similarly, for 3D-MobileNets from [here](https://drive.google.com/drive/folders/1eggpkmy_zjb62Xra6kQviLa67vzP_FR8). Save it in a directory, for example, `$HOME/datasets/PKUMMD/models/`

2. Download PKUMMD dataset

Please download RGB video data compressed in `.avi` format in 30 FPS from [here](https://drive.google.com/drive/folders/0B20a4UzO-OyMUVpHaWdGMFY1VDQ). Extract RGB JPEG frames and save it in, for example, `$HOME/datasets/PKUMMD/rgb_frames/<video_name>/image_00001.jpg`.

For the paper, the frames were saved in (171, 128) JPEG size format. That means, for model input size 224, the `Scale` transform will upscale the frame. Saving frames in higher resolution format and then downsampling it to fit the model input size might improve the performance of individual model. Note that our goal is not to optimize the performance of individual models but to show the performance of fusion method defined in the Clownfish paper.

### PKUMMD
You can use the following programs, 
* Convert from avi to jpg files using ```utils/video_jpg_pkummd.py```
```bash
python3 ./video_jpg_pkummd.py <avi_video_directory> <jpg_video_directory>
```

* Generate n_frames using ```utils/n_frames_pkummd.py```
```bash
python3 ./n_frames_pkummd.py <jpg_video_directory>
```

Assume/Create the structure of dataset directories in the following way:

```misc
~/
  dataset/
    PKUMMD/
      rgb_frames/
        0291-L/
        ...
      models/
      model_ckpt/
      splits/pkummd_cross_subject.json
      scores_dump/
```

## How to run scripts
Few examples are presented below,

1. Train an individual model, for example, ResNext-101 with batch_size 32
```bash
$ model="resnext-101" batch_size=32 run_type="train" ./run.sh
```

2. Dump scores for an individual model, for example,  Resnet-18 for validation (here, testing) videos 
```bash
$ model="resnet-18" run_type="predict" predict_type"val" ./run.sh
```

3. Train SimiNet model on the features of local model, Resnet-18 with checkpoint number 288
```bash
$ model="siminet" ckpt_num=288 ./run.sh
```

4. Train 3D-MobileNet for early discard
```bash
$ model="mobilenet-early-discard" ./run.sh 
```

For more options, please check the script file [./run.sh](./run.sh)
