# CUAS

This repository aims to compile valuable information, datasets, and trained models related to countering unmanned aerial systems.

## 1. Datasets

This section provides a brief overview of datasets available for training and evaluating models in the field of Counter Unmanned Aerial Systems (CUAS) detection. These datasets aim to support research and development efforts in identifying and countering unmanned aerial threats.

#### Dataset Statistics

Dataset | Size | Description | Links
--------|------|---------------------|-------
**MAV-VID** | *Training*: 29,500 image <br /> *Validation*: 10,732 images | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://bitbucket.org/alejodosr/mav-vid-dataset/src/master/)
**Drone-vs-Bird** | *Training*: 85,904 images <br /> *Validation*: 18,856 images | Comprises videos of UAV captured at long distances and surrounded by small objects | [Link](https://github.com/wosdetc/challenge/tree/master)
**Anti-UAV** | *Training*: 149,478 images <br /> *Validation*: 37,016 images | Contains RGB and IR recordings in different lightning and background conditions | [Link](https://anti-uav.github.io/dataset/)
**DUT Anti-UAV** | *Training*: 5,200 images <br /> *Validation*: 2,000 images <br /> 20 video clips | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://github.com/wangdongdut/DUT-Anti-UAV)
**Vis-Drone** | 288 video clips (261,908 frames) <br /> 10,209 static images | Drone-captured images of objects, such as pedestrians, cars, bicycles, and tricycles | [Link](https://github.com/VisDrone/VisDrone-Dataset)
**CUAS** | Total 8,555 images | Contains videos of drones captured from other drones and ground-based cameras | [Link](https://universe.roboflow.com/wk-meyzk/cuas-pq71v)


## 2. Trained Weights
[Google Drive](https://drive.google.com/drive/folders/1ZYfYUv00o63Q2O8Ozsd7JQif42KH38Ra?usp=drive_link)

Explore the following pre-trained detection models designed specifically for countering unmanned aerial systems. These models from the Google Drive are ready to be used or fine-tuned for CUAS detection tasks. They trained using various models, including YOLOv8 and Detection Transforms (DETR).

## 3. Challenges of Drone Detection and Tracking
* Out-of-View: Re-ID difficult when the target moves out of the frame.
* Occlusion: Target is partially or heavily occluded.
* Dynamic Background Clusters: Dynamic changes (e.g., buildings, leaves, birds) in the background around the target.
* Low Resolution: Especially when the area of the bounding box is small.
* Target Scale: Target usually occupies a small pixel area.
* Fast & Random Motion: Difficult to predict motion in next timestep.
* Moving Camera: Affects filters used for tracking.
* Limited Computational Resources: Limited by GPU and computing power on drone
