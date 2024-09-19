import cv2
import numpy as np
import datetime, argparse, os, re
from pyCFTrackers.lib.utils import APCE,PSR
from pyCFTrackers.lib.bbox_helper import cxy_wh_2_rect
from pyCFTrackers.cftracker.csrdcf import CSRDCF
from pyCFTrackers.cftracker.config import csrdcf_config


from util.params import *
from util.helper import *
from util.Generate_pm_pa import *
from util.Extract_Patch import *
from util.Detect_Patch import *

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
color_gt = color.copy()
prepreFrame = np.float32(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))
height, width, channel = color.shape

## Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = cam.get(cv2.CAP_PROP_FPS)        
output = cv2.VideoWriter(video_output_path+VIDEO_NAME+'.mov', fourcc,FPS,(width,height))   

######Background Subtraction #################
start = datetime.datetime.now()
if YOLO_MODEL:# Run YOLOv8 inference on the frame
    results = yolo_model.track(color,persist=True,verbose=False)

    # Get the boxes
    if results:
        yolo_coords = results[0].boxes.xywh.cpu().int().tolist()
end = datetime.datetime.now()
yolo_time = (end - start).total_seconds()
print(f"YOLOv8 Inference Time: {yolo_time:.2f}s")
print(f"YOLOv8 detected {len(yolo_coords)} targets. Passing bound box(es) to tracker...")
## ======== End of code from detect.py ============

yolo_coords = [cxy_wh_2_rect(coords[:2], coords[2:]) for coords in yolo_coords]
trackers = [CSRDCF(config=csrdcf_config.CSRDCFConfig()) for _ in range(len(yolo_coords))]
for yolo_coord, tracker in zip(yolo_coords, trackers):    
    tracker.init(color, yolo_coord)
while True:
    is_read_success, current_frame = cam.read()

    if not is_read_success:
        frameidx+=1
        break

    for tracker in trackers:
        x1, y1, w, h = tracker.update(current_frame,vis=True)
        if len(current_frame.shape)==2:
            current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
        score = tracker.score
        apce = APCE(score)
        psr = PSR(score)
        F_max = np.max(score)
        size = tracker.crop_size
        score = cv2.resize(score, size)
        score -= score.min()
        score /= score.max()
        score = (score * 255).astype(np.uint8)
        # score = 255 - score
        score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        center = (int(x1+w/2),int(y1+h/2))
        x0,y0=center
        x0=np.clip(x0,0,width-1)
        y0=np.clip(y0,0,height-1)
        center=(x0,y0)
        xmin = int(center[0]) - size[0] // 2
        xmax = int(center[0]) + size[0] // 2 + size[0] % 2
        ymin = int(center[1]) - size[1] // 2
        ymax = int(center[1]) + size[1] // 2 + size[1] % 2
        left = abs(xmin) if xmin < 0 else 0
        xmin = 0 if xmin < 0 else xmin
        right = width - xmax
        xmax = width if right < 0 else xmax
        right = size[0] + right if right < 0 else size[0]
        top = abs(ymin) if ymin < 0 else 0
        ymin = 0 if ymin < 0 else ymin
        down = height - ymax
        ymax = height if down < 0 else ymax
        down = size[1] + down if down < 0 else size[1]
        score = score[top:down, left:right]
        crop_img = current_frame[ymin:ymax, xmin:xmax]
        score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
        current_frame[ymin:ymax, xmin:xmax] = score_map
        show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),1)

        cv2.imshow('CSR-DCF Tracker', show_frame)
        if output is not None:
            output.write(show_frame)
        cv2.waitKey(1)


