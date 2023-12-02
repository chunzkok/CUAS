# CUAS

This repository aims to compile valuable information, datasets, and trained models related to countering unmanned aerial systems.

### 1. Datasets

#### Multirotor Aerial Vehicle VID (MAV-VID)

This dataset consists on videos at different setups of single UAV. 
It contains videos captured from other drones, ground based surveillance cameras and handheld mobile devices.
It can be downloaded in its [kaggle site](https://www.kaggle.com/alejodosr/multirotor-aerial-vehicle-vid-mavvid-dataset). 

#### Drone-vs-Bird
As part of the [International Workshop on Small-Drone Surveillance, Detection and Counteraction techniques](https://wosdetc2020.wordpress.com/drone-vs-bird-detection-challenge/)
of IEEE AVSS 2020, the main goal of this challenge is to reduce the high false positive rates that vision-based methods 
usually suffer. This dataset comprises videos of UAV captured at long distances and often surrounded by small objects, such as birds.

The videos can be downloaded upon request and the annotations can be downloaded via their [GitHub site](https://github.com/wosdetc/challenge).
The annotations follow a custom format, where a a .txt file is given for each video. Each annotation file has a line
for each video frame and the annotation is given in the format `<Frame number> <Number of Objects> <x> <y> <width> <height> [<x> <y> ...]`.

#### Anti-UAV
This multi-modal dataset comprises fully-annotated RGB and IR unaligned videos. Anti-UAV dataset is intended to provide 
a real-case benchmark for evaluating object tracker algorithms in the context of UAV. It contains recordings of 6 UAV 
models flying at different lightning and background conditions. This dataset can be downloaded in their [website](https://anti-uav.github.io/dataset/).

#### Dataset Statistics

Dataset | Size | Description | Links
--------|------|---------------------|-------
**MAV-VID** | *Training*: 29,500 image <br /> *Validation*: 10,732 images | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://bitbucket.org/alejodosr/mav-vid-dataset/src/master/)
**Drone-vs-Bird** | *Training*: 85,904 images <br /> *Validation*: 18,856 images | Comprises videos of UAV captured at long distances and surrounded by small objects | [Link](https://github.com/wosdetc/challenge/tree/master)
**Anti-UAV** | *Training*: 149,478 images <br /> *Validation*: 37,016 images | Contains RGB and IR recordings in different lightning and background conditions | [Link](https://anti-uav.github.io/dataset/)
**DUT Anti-UAV** | *Training*: 5,200 images <br /> *Validation*: 2,000 images <br /> 20 video clips | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://github.com/wangdongdut/DUT-Anti-UAV)
**Vis-Drone** | 288 video clips (261,908 frames) <br /> 10,209 static images | Drone-captured images of objects, such as pedestrians, cars, bicycles, and tricycles | [Link](https://github.com/VisDrone/VisDrone-Dataset)
**CUAS** | Total 8,555 images | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://universe.roboflow.com/wk-meyzk/cuas-pq71v)


### 2. Trained Weights
[Google Drive](https://drive.google.com/drive/folders/1ZYfYUv00o63Q2O8Ozsd7JQif42KH38Ra?usp=drive_link)

This Google Drive contains pre-trained weights from various models, including YOLOv8 and Detection Transforms (DETR).
