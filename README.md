# cuas

This repository aims to compile valuable information, datasets, and trained models related to countering unmanned aerial systems.

### Datasets

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

Dataset object size

Dataset | Size | Average Object Size
--------|------|---------------------
**MAV-VID** | *Training*: 53 videos (29,500 images) <br /> *Validation*: 11 videos (10,732 images) | 215 x 128 pxs (3.28% of image size)
**Drone-vs-Bird** | *Training*: 61 videos (85,904 images) <br /> *Validation*: 16 videos (18,856 images) | 34 x 23 pxs (0.10% of image size)
**Anti-UAV** | *Training*: 60 videos (149,478 images) <br /> *Validation*: 40 videos (37,016 images) | *RGB*: 125 x 59 pxs (0.40% image size)<br />*IR*: 52 x 29 pxs (0.50% image size)

Location, size and image composition statistics

### Trained Weights
[Google Drive](https://drive.google.com/drive/folders/1ZYfYUv00o63Q2O8Ozsd7JQif42KH38Ra?usp=drive_link)

This Google Drive contains pre-trained weights from various models, including YOLOv8 and Detection Transforms (DETR).
