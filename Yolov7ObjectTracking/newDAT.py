import os
import cv2
import time
import torch
import argparse
import tqdm
import numpy as np
import math
import json

from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(1, 'Yolov7ObjectTracking/')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *

#My library --------------------------------------
from CameraCabliration import *
from ObjectSpeedEstimate import *
from turn_detector import TurnDetector #update turn detector
from mvextractor.videocap import VideoCap #update turn detector

def object_dict(id, cls, X, Y, Z, speed, appear, x, y):
  if (speed < 0):
    speed = math.nan
  new_dict = {
    "id" : int(id),
    "class" : int(cls),
    "object_location" : [{
        "X" : round(float(X),2),
        "Y" : round(float(Y),2),
        "Z" : round(float(Z),2),
        'x' : x,
        'y' : y
    }],
    "velocity" : round(speed,2),
    "appear" : appear
  }
  return new_dict

#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img
#..............................................................................

def detect(video_url, vnp, speed, json_file_path, img_shape = (720,1280), ins_matrix_info = [[110, 70], 2.0], fps = 11, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= \
        video_url, 'Yolov7ObjectTracking/yolov7.pt', False, False, 640, True, True, False, False
    
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        # ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 

    #camera instri
    # Directories
    save_dir = Path(increment_path(Path('runs/detect') / 'object_tracking', exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    print(weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    #Dictionary
    pp_json = {}
    
    #Turn angle
    anglefile = open('angleann.txt', 'a')

    #OBJECT LOCATION TRAJECTORY -----------
    tracked = []
    object_track = []
    idx = -1
    traj_step = 5
    predict_step = 11
    object_track.append(Object(fps, id = 0))
    model_path = 'svm_model.pkl'
    predictor = Trajectory(model_path)
    #-------- -----------------------------

    #Camera calibration -----------------------------------------------
    CameraHeight = ins_matrix_info[1]
    FOV = (ins_matrix_info[0][0], ins_matrix_info[0][1])
    camera_calibration = ObjectClibration(img_shape[1], img_shape[0], FOV)
    intrinsic_mat = camera_calibration.get_intrinsic_matrix()
    
    
    #update turn detector
    turn_detector = TurnDetector(intrinsic_mat) #update turn detector
    video_mv_cap = VideoCap()
    video_mv_cap.open(source)
    #-----------------------------------
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]

        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]
        t2 = time_synchronized()
        

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)
        t3 = time_synchronized()
        
        #update turn detector
        # turn_angle = 0
        # flag, imgcap, motion_vector, _, _ = video_mv_cap.read()
        # turn_angle = turn_detector.process(imgcap, motion_vector)
        #--------------------------------------------------------

        # Apply Classifier
        
        

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            idx = idx + 1
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            f_dict = []
            obj_list = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                # tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                txt_str = ""
                # ego_car location ---------------------------------------------------------------------------------------
                ego_car = uv_to_world((img_shape[1]//2,img_shape[0]) , CameraHeight, vnp[idx], img_shape, FOV)
                txt_str += "%i %i %i %i %i %i %i %f %i" % (0, 0, 0, 2, 0, img_shape[1]//2, img_shape[0], speed[idx], -1)
                id, cls, X, Y, Z, x, y, spd, appear = 0, 0, 0, 2, 0, img_shape[1]//2, img_shape[0], speed[idx], -1
                f_dict.append(object_dict(int(id), int(cls), float(X), float(Y), float(Z), float(spd), int(appear),x,y))                    
                # =========================================================================================================
                txt_str += "\n"
                
                for track in tracks:                    
                    object_coor = (track.centroidarr[-1][0],track.centroidarr[-1][1])
                    coors = uv_to_world(object_coor, CameraHeight, vnp[idx], img_shape, FOV)
                    objvec = -1.0
                    appear_step = 1

                    # object trajectory (location for each object)------------------------
                    if (track.id+1) in tracked:
                        location = (coors[2]-ego_car[2], coors[0]-ego_car[0])
                        # print(track.id+1, 'in', tracked, ' is', (track.id+1) in tracked)
                        object_track[track.id+1].update_his(location)
                        objvec = object_track[track.id+1].speed_predict(speed[idx])
                        appear_step = object_track[track.id+1].FrameAppear()
                    else:
                        tracked.append(track.id+1)
                        location = (coors[2]-ego_car[2], coors[0]-ego_car[0])
                        object_track.append(Object(fps, id = track.id+1))
                        object_track[track.id+1].update_his(location)
                        appear_step = object_track[track.id+1].FrameAppear()
                    # ====================================================================
                    
                    # take information with each object -------------------------------------------------
                    id, cls, X, Y, Z, x, y, spd, appear = \
                        track.id+1, track.detclass, coors[0]-ego_car[0], coors[1], coors[2]-ego_car[2], \
                            track.centroidarr[-1][0],track.centroidarr[-1][1], objvec, appear_step
                    
                    if (appear >= 5 and cls < 9): obj_list.append([id, X, Z]) 
                    # ====================================================================================
                    
                    
                    # export meta data  
                    f_dict.append(object_dict(int(id), int(cls), float(X), float(Y), float(Z), float(spd), int(appear),x,y))

                    # if save_txt and not save_with_object_id:
                    #     # Normalize coordinates
                    #     txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                    #     txt_str += "\n"
                
                predict_obj = [index[0] for index in obj_list]
                risk3, risk5, risk10 = predictor.PredictRisk(idx, predict_obj, traj_step, predict_step, object_track, speed[idx])

                # if len(tracked_dets)>0:
                #     bbox_xyxy = tracked_dets[:,:4]
                #     identities = tracked_dets[:, 8]
                #     categories = tracked_dets[:, 4]
                #     draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
            
            # else: #SORT should be updated even with no detections
                # tracked_dets = sort_tracker.update()
            #........................................................
            
            json_step_name = f'frame{idx}' 
            frame_info = {
              'Object' : f_dict,
              'Risk_Object_3s' : risk3,
              'Risk_Object_5s' : risk5,
              'Risk_Object_10s' : risk10,
              'Turn_angle' : 0
            }
            pp_json[json_step_name] = [frame_info]
    
    final_json = {
        "Video":  [{
                    "Height" : 720,
                    "Width" : 1280,
                    "FrameCount" : idx
                  }],
        "CameraFOV": [{
                    "Horizontal" : 110,
                    "Vertical" : 70
                      }],
        "FrameInfo": {}
    }
    final_json["FrameInfo"] = [pp_json]
    json_save_path = json_file_path

    with open(json_save_path, 'w') as f:
      json.dump(final_json, f)

def TrajectoryAndMakingVideo(video_url, vnp_path, veclocity_path, json_file_path, fps, img_shape, ins_matrix_info):
    
    print('Video source: ', video_url)
    
    frame_count = 0
    
    f = open(vnp_path, 'r')
    
    vnp = []
    for i in f:
      x = i[:-1].split(",")
      vnp.append([float(x[0]), float(x[1])])
    vnp = np.array(vnp)

    f = open(veclocity_path, 'r')
    speed_ = []
    for i in f:
      x = float(i)
      speed_.append(x)
    speed = np.array(speed_)
    last = speed[-1]
    
    while (len(speed) < len(vnp)):
      speed = np.append(speed, last)

    weights = 'Yolov7ObjectTracking/yolov7.pt'
    update = True
    
    # if not os.path.exists(''.join(weights)):
    #     print('Model weights not found. Attempting to download now...')
    #     download('./')

    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            for weights in ['Yolov7ObjectTracking/yolov7.pt']:
                detect(video_url, vnp, speed, json_file_path, img_shape, ins_matrix_info, fps)
        else:
            detect(video_url, vnp, speed, json_file_path, img_shape, ins_matrix_info, fps)