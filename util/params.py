## FILE PATHS
from ultralytics import YOLO
import os
import cv2
from util.helper import initialize_params

YOLO_MODEL = True
FEATURE_TRACKER = True

### VIDEO STREAM OUTPUT ###
gt_text = False ### True if ground truth txt is available
video_output_path  = './results/video/'
txt_output_path =  './results/txt/'
if not os.path.exists(video_output_path):
    print ("path doesn't exist. trying to make")
    os.makedirs(video_output_path)
if not os.path.exists(txt_output_path):
    print ("path doesn't exist. trying to make")
    os.makedirs(txt_output_path)
    
### YOLO MODEL ###
yolo_path = './models/yolov8s.pt'
yolo_model = YOLO(yolo_path)


### MOTION MODEL ###
app_model_path = './models/max500_1_10_threelayers/'
app_model_path_track = './models/Appearance_OriImage/'
mvmodel_path = './models/motion/'
bimodel_path = './models/Adaboost/'
bimodel_path_track = './models/Adaboost_track/'
trackwinS = 15 #defult is 15, 2 

### PARAMETERS ###
THRESHOLD = 50 # threshold for the matching bounding box
A = 0.001
B = 50
all_params = initialize_params(A, B)
uplefty = all_params['uplefty']
downrighty = all_params['downrighty']
upleftx = all_params['upleftx']
downrightx = all_params['downrightx']
fileName = all_params['fileName']
debug = all_params['debug']
K = all_params['K']
qualityini = all_params['qualityini']
MaxCorns = all_params['MaxCorns']
mindist1 = all_params['mindist1']
use_ransac = all_params['use_ransac']
track_len = all_params['track_len']
lamda = all_params['lamda']
quality = all_params['quality']
maxcorners = all_params['maxcorners']
mindist = all_params['mindist']
detect_interval = all_params['detect_interval']
feature_params = {
        'maxCorners': all_params['MaxCorns'],
        'qualityLevel': all_params['qualityini'],
        'minDistance': all_params['mindist1'],
        'blockSize': 5
    }

lk_params = {
        'winSize': (19, 19),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    }
## parameter for feature detection(error image) and tracking[for feature extraction and tracking]
feature_params_track = dict( maxCorners = 500,
                    qualityLevel = quality/20.0,
                    minDistance = mindist,
                    blockSize = 9 )

feature_params_track_oriImg = feature_params_track
                        
lk_params_track = dict( winSize  = (19, 19),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                minEigThreshold=1e-4)

lk_params_track_ori = dict( winSize  = (25, 25),
                maxLevel = 3,
                flags = cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                minEigThreshold=1e-4)

feature_params_Detect = dict( maxCorners = 10,
                        qualityLevel = 0.00000015,
                        minDistance = 0,
                        blockSize = 3 )
