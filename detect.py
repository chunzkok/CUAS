from numpy import newaxis
import datetime, argparse, os, re
from collections import defaultdict

from util.params import *
from util.helper import *
from util.Generate_pm_pa import *
from util.Extract_Patch import *
from util.Detect_Patch import *

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
h, w, channel = color.shape

## Create output txt file
outputFeature = "time_layer: "+ str(frameidx)+" detections: "
f_txt = open(txt_output_path+ VIDEO_NAME+'.txt','w')
f_txt.write('frame_idx, UAV_ID, min_x, min_y, max_x, max_y\n')

## Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = cam.get(cv2.CAP_PROP_FPS)        
output = cv2.VideoWriter(video_output_path+VIDEO_NAME+'.mov', fourcc,FPS,(w,h))   

maxD=4
ind = 1
radius = 10
pImg, H_back = None, None
appmodel, appmodel_track, mvmodel, combinemodel, combinemodel_track = load_models(app_model_path, app_model_path_track, mvmodel_path, bimodel_path, bimodel_path_track, ind)

# read in Xt-1
color=cam.read()[1]
color_gt =color.copy()
frameidx+=1
Xtminus1 = np.float32(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))
blocks = np.ones((h,w), dtype='float32')

Dotft = []
Patchft=[]
maxPatchId = 0
# Store the track history
track_history = defaultdict(lambda: [])
combined_id = 0
tracker = (0,0)
ft_coords, ft_ids = [], []

while True:
    yolo_boxes = []
    feature_boxes = []

    # List to keep track of matched coordinates
    matched_coords = []

    ######Background Subtraction #################

    print('frameID:', frameidx)
    gray = Xtminus1.copy()

    # read in current frame Xt
    future_color = cam.read()[1]
    if future_color is None:
        frameidx+=1
        f_txt.write(f"{frameidx}, \n")
        break

    frameidx+=1
    start = datetime.datetime.now()

    if YOLO_MODEL:# Run YOLOv8 inference on the frame
        # results = yolo_model.predict(future_color,max_det=1)
        results = yolo_model.track(future_color,persist=True,verbose=False)

        # Get the boxes and track IDs
        if results:
            yolo_coords = results[0].boxes.xywh.cpu().int().tolist()
            yolo_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            yolo_boxes = zip(yolo_coords, yolo_ids)
    end = datetime.datetime.now()
    yolo_time = (end - start).total_seconds()
    print(f"YOLOv8 Inference Time: {yolo_time:.2f}s")

    start = datetime.datetime.now()
    if FEATURE_TRACKER:
        Xt = np.float32(cv2.cvtColor(future_color, cv2.COLOR_RGB2GRAY))
        color_gt = color.copy()
        oriImage = color_gt.copy()
        oriImage_1 = color_gt.copy()
    
        # extract feature points for previous frame gray = Xt-1. By using maskOut function, only keep features outside the pitot tube area
        if pImg is None or frameidx % track_len == 0:
            pImg = cv2.goodFeaturesToTrack(np.uint8(gray), **feature_params)
            # pImg = maskOut(blocks, pImg)

        # compute onedirectional error Et-1 using backward transform to save computational time
        if (frameidx) % detect_interval == 0:
            weightedError,H_back,pImg = backgroundsubtraction(gray, prepreFrame, Xt, pImg, blocks, lamda, lk_params, use_ransac)
        else:
            H_back,pImg = backgroundMotion(gray, prepreFrame, Xt, pImg, blocks, lamda, lk_params, use_ransac)
        
        ###########################Feature Extraction on Background Subtracted Image for Every Other 20 Frames###############################

        if len(Dotft)>0:
            d1, d, p1, pPers, p0, st1, ft_mv, ft_app, gt_labels = generatePatches_MV_trackV1(Dotft, gray, Xt, H_back, lk_params_track_ori, radius,w, h, color, np.zeros_like(Xt))
            score_mv = mvmodel.predict(ft_mv, batch_size = 1000000)
            score_app = appmodel_track.predict(ft_app,batch_size= 2560)
            bifeature = np.hstack([score_app[:,0].reshape(-1,1),score_mv[:,0].reshape(-1,1)])                    
            trst = combinemodel_track.predict(bifeature)                
            Dotft,indrm = dotupdate(Dotft, Patchft)
            oriImage = visDotft(oriImage, Dotft,w, h)
            Dotft = dottrack_detect(Dotft, p1[indrm], pPers[indrm], trst[indrm], st1[indrm], d1[indrm], d[indrm], Patchft)
            oriImage = visPtV1(oriImage, p0, st1, d1)
            
        if len(Patchft)>0:
            oriImage, ft_coords, ft_ids = visDetect_Kalman(Patchft, oriImage, radius, w, h)
            outputFeature = writeDetect(outputFeature, radius, Patchft, w, h)
            Patchft = patch_KalmanTracking(Dotft, Patchft, H_back, w, h)

        if (frameidx) % detect_interval == 0:
            mv, detectedPatches, errorPatches, gt_labels, detectedLocs, curLocslll, hit, ftNo, FAno = Extract_Patch(frameidx, gray, Xt, weightedError, [], H_back, feature_params_track, feature_params_track_oriImg, lk_params_track, radius, color, future_color, np.zeros_like(Xt), oriImage)
            
            if mv.shape[0]>0:
                errorPatches = errorPatches[:,:,:, newaxis]
                mv = np.hstack([mv[:,4:6],mv[:,10:]])
                data_np_test = np.concatenate([detectedPatches/255.0 ,errorPatches/255.0,errorPatches/255.0,errorPatches/255.0], axis=3)
                test_output_app = appmodel.predict(data_np_test,batch_size= 2560)
                test_output_mv = mvmodel.predict(mv, batch_size = 1000000)
                mvmafeature = np.hstack([test_output_app[:,0].reshape(-1,1),test_output_mv[:,0].reshape(-1,1)])
                dt_lable = combinemodel.predict(mvmafeature)                   
                oriImage = visPosPatch_Kalman(dt_lable, gt_labels, detectedLocs, oriImage, radius)
                oriImage, Dotft, Patchft, maxPatchId = DetectOnX_V2(maxD, maxPatchId, oriImage, gray, Xt, lk_params_track_ori, H_back, detectedLocs, curLocslll, dt_lable, detectedPatches,feature_params_Detect, radius, Dotft, Patchft)

    end = datetime.datetime.now()
    ft_time = (end - start).total_seconds()
    print(f"Feature Tracker Time: {ft_time:.2f}s")
    total_time = yolo_time + ft_time
    fps = f"FPS: {1 / total_time:.2f}"

    # fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    draw_str(oriImage, 20, 20, fps)
    # oriImage = results[0].plot(img=oriImage)

    # Iterate through coordinates from both trackers
    if ft_coords and yolo_coords:
        for ft_coord, ft_id in zip(ft_coords, ft_ids):
            for yolo_coord, yolo_id in zip(yolo_coords, yolo_ids):
                # Check if coordinates are close
                if is_close(ft_coord, yolo_coord, THRESHOLD):
                    # Combine bounding boxes
                    combined_box = combine_bounding_boxes((ft_coord[0], ft_coord[1], 0, 0), yolo_coord) # (minx, miny, maxx, maxy)
                    combined_id, tracker = update_tracking_id(ft_id, yolo_id, tracker, combined_id)
                    matched_coords.append((ft_coord, ft_id, yolo_coord, yolo_id, combined_box, combined_id))

    # Draw combined bounding boxes in green
    detections = ""
    for ft_coord, ft_id, yolo_coord, yolo_id, combined_box, combined_id in matched_coords:
        cv2.rectangle(oriImage, (combined_box[0], combined_box[1]), (combined_box[2], combined_box[3]), (0, 255, 0), 2)
        draw_str_large(oriImage, combined_box[0], combined_box[1] - 50, f"UAV ID: {combined_id}")
        detections += f"{combined_id}, {combined_box[0]}, {combined_box[1]}, {combined_box[2]}, {combined_box[3]}"
        draw_str(oriImage, combined_box[0], combined_box[1] - 30, f"FT ID: {ft_id}")
        draw_str(oriImage, combined_box[0], combined_box[1] - 10, f"YOLO ID: {yolo_id}")
    f_txt.write(f'{frameidx}, {detections}\n')
    
    # Draw feature tracker coordinates in blue
    for ft_coord, ft_id in zip(ft_coords, ft_ids):
        # print(ft_coord, ft_id)
        # Check if this coordinate is part of a matched pair
        if not any(ft_coord == match[0] for match in matched_coords):
            cv2.rectangle(oriImage, (ft_coord[0] - radius, ft_coord[1] - radius), (ft_coord[0] + radius, ft_coord[1] + radius), (255, 0, 0), 2)
            draw_str(oriImage, ft_coord[0] - radius, ft_coord[1] - radius - 10, f"FT ID: {ft_id}")

    # Draw YOLO tracker coordinates in red and plot tracks
    for yolo_coord, yolo_id in zip(yolo_coords, yolo_ids):
        # Check if this coordinate is part of a matched pair
        if not any(yolo_coord == match[2] for match in matched_coords):
            cv2.rectangle(oriImage, (yolo_coord[0]-yolo_coord[2]//2, yolo_coord[1]-yolo_coord[3]//2),
                         (yolo_coord[0] + yolo_coord[2]//2, yolo_coord[1] + yolo_coord[3]//2), (0, 0, 255), 2)
            draw_str(oriImage, yolo_coord[0]-yolo_coord[2]//2, yolo_coord[1]-yolo_coord[3]//2 - 10, f"YOLO ID: {yolo_id}")

        # Draw the tracking lines
        X,Y,W,H = yolo_coord
        track = track_history[yolo_id]
        track.append((float(X), float(Y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(oriImage, [points], isClosed=False, color=(230, 230, 230), thickness=1)

    output.write(oriImage)
    prepreFrame = Xtminus1.copy()
    color = future_color.copy()
    Xtminus1 = Xt.copy()
    future_color_gt = future_color.copy()

output.release()
f_txt.close()




