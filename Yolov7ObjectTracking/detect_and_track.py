import os
import cv2
import time
import torch
import argparse
import tqdm
import yaml
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(1, 'Yolov7ObjectTracking/')

from CameraCabliration import *
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

from ObjectSpeedEstimate import *
from turn_detector import TurnDetector

import numpy as np
import math
import json

#For SORT tracking
from sort import *
import skimage

from mvextractor.videocap import VideoCap

def parse_config(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return data

#..................d............. Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, risk, predict_obj, img_shape, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(j) for j in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        id = int(identities[i]) if identities is not None else 0
        label = str(id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        if id in risk:
            X = [posx[1] for posx in predict_obj if posx[0]==id]
            Y = [posy[1] for posy in predict_obj if posy[0]==id]
            distance = np.sqrt(X[0]**2 + Y[0]**2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
            
            org = [img_shape[1]//2, img_shape[0]]
            color = (255, 0, 0)  # Màu xanh lá cây
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 1
            place = (int(0.5*(org[0]+0.5*(x1+x2))), int(0.5*(org[1]+y2)))
            text = f'{round(distance,2)}'
            cv2.putText(img, text, place, font, font_scale, color, thickness)

            cv2.line(img, (org[0], org[1]), (int((0.5*(x1+x2))), int(y2)), color, thickness)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
        
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
    return img
#..............................................................................

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

def detect(file_source, vnp, speed, json_file_path, img_shape = (720,1280), save_img=False):
    weights = 'Yolov7ObjectTracking/yolov7.pt'
    source = file_source
    save_txt = True
    imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= 640, True, True, False, True
    save_img = True and not source.endswith('.txt')  # save inference images
    video_url = source
    print(video_url)

    class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']

    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    #........Turn angle.......
    anglefile = open('angleann.txt', 'a')
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   
    # Directories
    save_dir = Path(increment_path(Path('runs/detect') / 'object_tracking', exist_ok=False))  # increment run
    save_dir = Path('Yolov7ObjectTracking')/save_dir
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
    print(save_dir)

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    #Dictionary
    #---------------------------------------------
    pp_json = {}
    #---------------------------------------------

    if trace:
        model = TracedModel(model, device, img_size = imgsz)

    if half:
        model.half()  # to FP16

    # Second-stage classifier

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1


    t0 = time.time()
    #Tracking -----------------------------
    tracked = []
    object_track = []
    fps = 11
    idx = -1
    traj_step = 5
    predict_step = 110
    object_track.append(Object(fps, id = 0))
    model_path = '/content/drive/MyDrive/ADAS/lstm11s.pth'
    predictor = Trajectory(model_path, n_steps=5, n_features=1)
    #-------- -----------------------------
    FOV = (110,70)
    
    camera_calibration = ObjectClibration(img_shape[1], img_shape[0], FOV)
    intrinsic_mat = camera_calibration.get_intrinsic_matrix()
    
    turn_detector = TurnDetector(intrinsic_mat)
    
    config_file = "Yolov7ObjectTracking/ProjectConfig.yaml"
    config = parse_config(config_file)
    
    bar = tqdm.tqdm(len(dataset))
    for path, img, im0s, vid_cap in dataset:
        bar.update(1)
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
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        
        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = non_max_suppression(pred, 0.25, 0.45)
        t3 = time_synchronized()
     
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            idx = idx + 1
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
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
                
                obj_box = []
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                    
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                txt_str = ""
                FOV = (110,70)
                img_shape = (720,1280)
                
                #camera height setting
                ego_car = uv_to_world((img_shape[1]//2,img_shape[0]) , 1.64, vnp[idx], img_shape, FOV)
                txt_str += "%i %i %i %i %i %i %i %f %i" % (0, 0, 0, 2, 0, img_shape[1]//2, img_shape[0], speed[idx], -1)
                id, cls, X, Y, Z, x, y, spd, appear = 0, 0, 0, 2, 0, img_shape[1]//2, img_shape[0], speed[idx], -1
                f_dict.append(object_dict(int(id), int(cls), float(X), float(Y), float(Z), float(spd), int(appear),x,y))
                
                txt_str += "\n"
                # print(len(object_track))
                #loop over tracks
              
                for track in tracks:
                    object_coor = (track.centroidarr[-1][0],track.centroidarr[-1][1])

                    coors = uv_to_world(object_coor, 2.0, vnp[idx], img_shape, FOV)
                    
                    objvec = -1.0
                    appear_step = 1
                    
                    # print('frame_id = ',idx)
                    # print(len(object_track))
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
      
                
                    id, cls, X, Y, Z, x, y, spd, appear = track.id+1, track.detclass, coors[0]-ego_car[0], coors[1], coors[2]-ego_car[2], track.centroidarr[-1][0],track.centroidarr[-1][1], objvec, appear_step
                    
                    if (appear >= 5 and cls < 9):
                        obj_list.append([id, X, Z])
                    
                    # print()
                    f_dict.append(object_dict(int(id), int(cls), float(X), float(Y), float(Z), float(spd), int(appear),x,y))
                
                predict_obj = [index[0] for index in obj_list]
                risk3, risk5, risk10 = predictor.PredictRisk(idx, predict_obj, traj_step, predict_step, object_track, speed[idx])
                # draw boxes for visualization
                
                # if len(tracked_dets)>0:
                #     bbox_xyxy = tracked_dets[:,:4]
                #     identities = tracked_dets[:, 8]
                #     categories = tracked_dets[:, 4]
                
                #     cv2.imwrite(f'tam/{idx}.jpg', draw_boxes(im0, bbox_xyxy, risk3, obj_list, img_shape, identities))

            json_step_name = f'frame{idx}' 
            frame_info = {
              'Object' : f_dict,
              'Risk_Object_3s' : risk3,
              'Risk_Object_5s' : risk5,
              'Risk_Object_10s' : risk10
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

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    
    print(f'Done. ({time.time() - t0:.3f}s)')
    
    json_save_path = json_file_path
    with open(json_save_path, 'w') as f:
      json.dump(final_json, f)


def process(video_path, vnp_path, veclocity_path, json_save_path, fps, img_shape):
    '''
      '/content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/vnp.txt'
      /content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/velocity.txt'
    '''
    print('Video source: ', video_path)
    
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

    print(len(speed))


    with torch.no_grad():
        detect(video_path, vnp, speed, json_save_path, img_shape)    

def TrajectoryAndMakingVideo(video_path, vnp_path, veclocity_path, json_file_path, fps, img_shape):
    process(video_path, vnp_path, veclocity_path, json_file_path, fps, img_shape)

if __name__ == '__main__':
    video_path = '/content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/20211213110138_0_8.avi'
    vnp_path = '/content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/vnp.txt'
    veclocity_path = '/content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/velocity.txt'
    json_file_path = 'data.json'
    fps = 11
    process(video_path, vnp_path, veclocity_path, json_file_path, fps)
    
