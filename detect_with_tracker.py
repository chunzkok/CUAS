import cv2
import numpy as np
import datetime, argparse, os, re, sys
from collections import deque
from pathlib import Path 

sys.path.insert(0, str(Path("./pyCFTrackers/").absolute()))
from pyCFTrackers.lib.bbox_helper import cxy_wh_2_rect, center2corner
from pyCFTrackers.cftracker.strcf import STRCF
from pyCFTrackers.cftracker.config import strdcf_hc_config

from util.params import *
from util.helper import *
from util.Target import Target
from util.Generate_pm_pa import *
from util.Extract_Patch import *
from util.Detect_Patch import *

generate_tracker = lambda: STRCF(config=strdcf_hc_config.STRDCFHCConfig()) 

###############################################################################################


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
color = cam.read()[1]
height, width, channel = color.shape
Target.next_frame(color)

## Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = cam.get(cv2.CAP_PROP_FPS)        
output = cv2.VideoWriter(video_output_path+VIDEO_NAME+'.mov', fourcc,FPS,(width,height))   

is_complete = False
while not is_complete:
    is_read_success, current_frame = cam.read()
    Target.next_frame(current_frame)
    start = datetime.datetime.now()
    if YOLO_MODEL:# Run YOLOv8 inference on the frame
        results = yolo_model.track(current_frame,persist=True,verbose=False)

        # Get the boxes
        if results:
            yolo_coords = results[0].boxes.xywh.cpu().int().tolist()
            yolo_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            yolo_data = dict(zip(yolo_ids, yolo_coords))
    end = datetime.datetime.now()
    yolo_time = (end - start).total_seconds()
    print(f"YOLOv8 Inference Time: {yolo_time:.2f}s")
    ## ======== End of code from detect.py ============

    if yolo_data:
        Target.batch_update_yolo(yolo_data)
    
    show_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR) if len(current_frame.shape) == 2 else current_frame
    for target in Target.all_targets:
        sources, xywh = target.aggregate_coords()
        if xywh is None:
            continue
        x, y, w, h = xywh
        if "YOLO" in sources or "Depth detection" in sources:
            show_frame=cv2.rectangle(current_frame, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255),1)
        else:
            show_frame=cv2.rectangle(current_frame, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), (255, 0, 0),1)
        for i, source in enumerate(sources, start=1):
            draw_str(show_frame, int(x-w/2), int(y-h/2) - i*15, source)
        draw_str(show_frame, int(x-w/2), int(y+h/2)+15, f"ID: {target.id}")
    draw_str(show_frame, 20, 20, f"FPS: {1 / (datetime.datetime.now() - start).total_seconds():.2f}")
    
    cv2.imshow('Detection with STRCF Tracker', show_frame)
    if output is not None:
        output.write(show_frame)
    cv2.waitKey(1)