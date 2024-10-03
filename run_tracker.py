import cv2
import numpy as np
import datetime, argparse, os, re, sys
from pathlib import Path 
from collections import deque

sys.path.insert(0, str(Path("./pyCFTrackers/").absolute()))
from pyCFTrackers.lib.utils import APCE,PSR
from pyCFTrackers.lib.bbox_helper import cxy_wh_2_rect, center2corner
from pyCFTrackers.cftracker.strcf import STRCF
from pyCFTrackers.cftracker.config import strdcf_hc_config

from util.params import *
from util.helper import *
from util.Generate_pm_pa import *
from util.Extract_Patch import *
from util.Detect_Patch import *

generate_tracker = lambda: STRCF(config=strdcf_hc_config.STRDCFHCConfig()) 
REFRESH_INTERVAL = 90 # in frames

## ======== Taken from detect.py ============
# Argument parser setup
parser = argparse.ArgumentParser(description='Process video file.')
parser.add_argument('VIDEO_NAME', type=str, help='Path to the video file')
args = parser.parse_args()

# Get the input video file path
VIDEO_NAME = args.VIDEO_NAME

if str(VIDEO_NAME) == "0":
    print("Using Live Camera Feed")
    video_input = 0
else:
    print("Using Video File: ", VIDEO_NAME)
    video_input = './data/videos/'+ VIDEO_NAME
    
    # Check if the file exists
    if not os.path.isfile(video_input):
        print(f"The file {video_input} does not exist.")
        exit()
    
    VIDEO_NAME = re.sub(r'\.[a-zA-Z0-9]+$', '', VIDEO_NAME)
    
# Read Camera input
cam = cv2.VideoCapture(video_input)

# read in first frame to obtain the video channels info
frameidx = 1
color = cam.read()[1]
height, width, channel = color.shape

## Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = cam.get(cv2.CAP_PROP_FPS)        
output = cv2.VideoWriter(video_output_path+VIDEO_NAME+'.mov', fourcc,FPS,(width,height))   

is_complete = False
max_num_targets = 0
trackers = deque([])
while not is_complete:
    is_read_success, current_frame = cam.read()
    start = datetime.datetime.now()
    if YOLO_MODEL:# Run YOLOv8 inference on the frame
        results = yolo_model.track(current_frame,persist=True,verbose=False)

        # Get the boxes
        if results:
            yolo_coords = results[0].boxes.xywh.cpu().int().tolist()
    end = datetime.datetime.now()
    yolo_time = (end - start).total_seconds()
    print(f"YOLOv8 Inference Time: {yolo_time:.2f}s")
    ## ======== End of code from detect.py ============

    if yolo_coords:
        print(f"YOLOv8 detected {len(yolo_coords)} targets. Passing bound box(es) to tracker...")
        max_num_targets = max(len(yolo_coords), max_num_targets)

        new_trackers = [generate_tracker() for _ in range(len(yolo_coords))]
        for yolo_coord, tracker in zip(yolo_coords, new_trackers):    
            tracker.init(current_frame, cxy_wh_2_rect(yolo_coord[:2], yolo_coord[2:]))

        trackers.extend(new_trackers)
        while len(trackers) > max_num_targets:
            trackers.popleft()
    else:
        print(f"YOLOv8 did not detect any targets. Trying next frame...")
        continue
    for _ in range(REFRESH_INTERVAL):
        is_read_success, current_frame = cam.read()

        if not is_read_success:
            frameidx+=1
            is_complete = True
            break

        for tracker in trackers:
            x1, y1, w, h = tracker.update(current_frame,vis=True)
            if len(current_frame.shape)==2:
                current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
            show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),1)
            if YOLO_MODEL:# Run YOLOv8 inference on the frame
                results = yolo_model.track(current_frame,persist=True,verbose=False)

                # Get the boxes
                if results:
                    for coord in results[0].boxes.xywh.cpu().int().tolist():
                        x1, y1, x2, y2 = center2corner(coord)
                        show_frame=cv2.rectangle(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),1)

            cv2.imshow('STRCF Tracker', show_frame)
            if output is not None:
                output.write(show_frame)
            cv2.waitKey(1)


