import argparse
import os
import random
import time
from os.path import isfile, join, split
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import tqdm
import yaml
import cv2

from torch.optim import lr_scheduler
from DeepHough.logger import Logger

from DeepHough.dataloader import get_loader
from DeepHough.model.network import Net
from skimage.measure import label, regionprops
from DeepHough.utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point
import pandas as pd

config = "DeepHough/config.yml"
pretrain_path = "DeepHough/dht_r50_nkl_d97b97138.pth"

assert os.path.isfile(config)
CONFIGS = yaml.safe_load(open(config))

tmp = ""
# merge configs
if tmp != "" and tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

def openVNPpath(path):
  f = open(path, 'r')
  vnp = []
  for i in f:
    num1, num2 = map(float, i.strip().split(','))
    rounded_num1 = int(round(num1))
    rounded_num2 = int(round(num2))
    vnp.append([rounded_num1, rounded_num2])
  return vnp

def MaxRange(x, range=25):
  x_values = np.arange(min(x), max(x), range)
  bins = pd.cut(x, x_values)
  counts = bins.value_counts().sort_index()
  return (counts.idxmax().left + counts.idxmax().right)*0.5

def fix_outliers_iqr(arr, ins, factor=1.5):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
    non_outliers_median = np.median(arr[(arr >= lower_bound) & (arr <= upper_bound)])
    # You can replace the outliers with a specific value or remove them:
    # Replace outliers with the median or mean of the non-outlier values:
    if (non_outliers_median == 0):
      non_outliers_median = ins
    arr_fixed = np.where((arr < lower_bound) | (arr > upper_bound) | (arr == 0), non_outliers_median, arr)

    return arr_fixed

def FixVNP(path):
  vnp = openVNPpath(path)

  x = [i[0] for i in vnp]
  y = [i[1] for i in vnp]

  x = np.array(x)
  y = np.array(y)

  vnps = []
  if (len(x) != 0):
    newx = fix_outliers_iqr(x, MaxRange(x), factor=1.5)
    newy = fix_outliers_iqr(y, MaxRange(y), factor=1.5)

    for i in range(len(newx)):
      vnps.append([int(newx[i]), int(newy[i])])

  return vnp, vnps

def VanishingPointDetection(output_path, video_path, frame_interval=1):

    # logger.info(args)

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
    model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    if pretrain_path:
        if isfile(pretrain_path):
            print("=> loading pretrained model")
            checkpoint = torch.load(pretrain_path)
            if 'state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print('No pretrained model found')
    # dataloader
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"], 
                                batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)

    print("Data loading done.")
    print("Start testing.")
    
    index, iter_num = test(test_loader, model, output_path, frame_interval, video_path)
    if index / iter_num < 0.5:
      print('********************************************************************\n')
      print(f'WARNING: just {index}/{iter_num} was detect')
      print('\n********************************************************************')
    else:
      print(index, ' VNPs were detected')
    print("Done!")

def test(test_loader, model, path_myf, frame_interval, video_path):
    # switch to evaluate mode
    model.eval()

    previous = []
    vn_point = (0,0)
    f = open(path_myf,'w')
    print("File Path: ", path_myf)
    iter_num = 0
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        iter_num = len(test_loader.dataset)
        ftime = 0
        ntime = 0
        index = 0
        for i, data in enumerate(bar):
            t = time.time()
            images, names, size = data
            
            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            # size = (size[0].item(), size[1].item())       
            key_points = model(images)
            
            key_points = torch.sigmoid(key_points)
            ftime += (time.time() - t)
            t = time.time()
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)

            size = (size[0][0], size[0][1])
            b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
            scale_w = size[1] / 400
            scale_h = size[0] / 400
            
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0] * scale_h))
                x1 = int(np.round(b_points[i][1] * scale_w))
                y2 = int(np.round(b_points[i][2] * scale_h))
                x2 = int(np.round(b_points[i][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1-y2) / (x1-x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)
            
            vn_point, previous, flag = vnp(b_points, width=CONFIGS["FRAME"]["WIDTH"], height=CONFIGS["FRAME"]["HEIGHT"], previous_lop=previous,\
                                     path=video_path, frame_iter=frame_interval, current_frame=index, fill_with_rvnp=False)
            # print(index)
            # print(join(visualize_save_path, names[0].split('/')[-1]), ' => ', vn_point, '\n')
            for i in range(frame_interval):
                # f.write(str(vn_point[0])+','+str(vn_point[1])+'\n')
                f.write(str(vn_point[0])+','+str(vn_point[1])+','+str(flag)+'\n')
            if flag == False:
              index += 1
            
            # plt.scatter(int(vn_point[0]), int(vn_point[1]), color='red', marker='o')

            # vis = visulize_mapping(b_points, size[::-1], names[0])
            #cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)
            np_data = np.array(b_points)
            np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]), np_data)

            # if CONFIGS["MODEL"]["EDGE_ALIGN"] and args.align:
            #     for i in range(len(b_points)):
            #         b_points[i] = edge_align(b_points[i], names[0], size, division=5)
            #     vis = visulize_mapping(b_points, size, names[0])
            #     #cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]+'_align.png'), vis)
            #     np_data = np.array(b_points)
            #     np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]+'_align'), np_data)
            bar.update(1)
            
    #print('forward time for total images: %.6f' % ftime)
    #print('post-processing time for total images: %.6f' % ntime)
    #return ftime + ntime
    return index, iter_num

def get_line(p1, p2):
    y1 = p1[0]
    x1 = p1[1]
    y2 = p2[0]
    x2 = p2[1]
    if (x1 == x2): return None
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a, b

def get_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2
    if m1 == m2:
        return None
    u_i = (c2 - c1) / (m1 - m2)
    v_i = m1*u_i + c1
    return u_i, v_i

def set_intersect(b_points):
    set_p = b_points
    line = []
    num_line = 0
    intersect_point = []
  
    for point2 in set_p:
        p1 = point2[:2]
        p2 = point2[2:]
        line.append(get_line(p1,p2))
        num_line = num_line + 1

    if (num_line == 0): 
        return None

    for i in range(num_line):
        for j in range(i+1,num_line):
            if (line[i] == None or line[j] == None): 
                continue
            intersect_point.append(get_intersection(line[i], line[j]))
    return intersect_point

def remove_outlier(inte_point, width, height):
  
    intersect_point = []

    if (inte_point == None):
        return intersect_point

    scale = CONFIGS["FRAME"]["SCALE"]
    cut_height = height/scale #config

    min_x = width/3
    max_x = 2*width/3
    min_y = height/3 - cut_height
    max_y = 2*height/3 - cut_height

    for points in inte_point:
        if (points == None):
            continue
        if (points[0] <= max_x and points[0] >= min_x):
            if (points[1] <= max_y and points[1] >= min_y):
                intersect_point.append(points)

    return intersect_point

def vnp(b_points, width, height, previous_lop, path, frame_iter=1, current_frame=0, fill_with_rvnp=False):
    p0 = 0
    p1 = 0
    itersect_point = set_intersect(b_points)
    flag = False
    
    scale = CONFIGS["FRAME"]["SCALE"]
    cut_height = height/scale #config

    ls_itersect_point = remove_outlier(itersect_point, width, height)

    next_previous_lop = []
   
    if (ls_itersect_point == []):
        flag = True
        p_0 = 0
        p_1 = 0
        for i in previous_lop:
            p_0 = p_0 + i[0]
            p_1 = p_1 + i[1]
            p0, p1 = (p_0/len(previous_lop),p_1/len(previous_lop)+cut_height)
        next_previous_lop = previous_lop
            
    
    if (ls_itersect_point != []):
        p_0 = 0
        p_1 = 0
        for i in ls_itersect_point:
            p_0 = p_0 + i[0]
            p_1 = p_1 + i[1]
            p0, p1 = (p_0/len(ls_itersect_point),p_1/len(ls_itersect_point)+cut_height)
        next_previous_lop = ls_itersect_point
    
    return (p0,p1), next_previous_lop, flag
   
if __name__ == '__main__':
    output_path = '/content/drive/MyDrive/ADAS/Runs/20211213110138_0_8/vnp.txt'
    VanishingPointDetection(output_path)
