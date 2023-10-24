# from setup import SettingEnvironment
# SettingEnvironment()
from pathlib import Path
from vnp import *
from Yolov7ObjectTracking.detect_and_track import *
from SpeedETool import *
import json
import cv2


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--folderpath', help='')
parser.add_argument('--videopath', help='')
parser.add_argument('--KML_file_path', help='')

parser.add_argument('--FOV_horizontal', type=int,  default=110, help='')
parser.add_argument('--FOV_vertical', type=int,  default=70, help='')
parser.add_argument('--CameraHeight', type=float,  default=2.0, help='')

parser.add_argument('--ImageWidth', type=float,  default=1280, help='')
parser.add_argument('--ImageHeight', type=float,  default=720, help='')
args = parser.parse_args()

def makeJson(frame_count, data, risk_json_path):
  datajson = {}
  f = open(data)
  data = json.load(f)

  for i in range(frame_count):  
    frame_info = []
    risk3s,  risk5s, risk10s = [], [], []

    risk_obj_3s = [int(obj_id) for obj_id in data['FrameInfo'][0][f'frame{i}'][0]['Risk_Object_3s']]
    for id in risk_obj_3s:
      for obj in data['FrameInfo'][0][f'frame{i}'][0]['Object']:
        if (obj['id'] == id):
          risk3s.append({'id':id, 'object_location':obj['object_location']})
    
    risk_obj_5s = [int(obj_id) for obj_id in data['FrameInfo'][0][f'frame{i}'][0]['Risk_Object_5s']]
    for id in risk_obj_5s:
      for obj in data['FrameInfo'][0][f'frame{i}'][0]['Object']:
        if (obj['id'] == id):
          risk5s.append({'id':id, 'object_location':obj['object_location']})
    
    risk_obj_10s = [int(obj_id) for obj_id in data['FrameInfo'][0][f'frame{i}'][0]['Risk_Object_10s']]
    for id in risk_obj_10s:
      for obj in data['FrameInfo'][0][f'frame{i}'][0]['Object']:
        if (obj['id'] == id):
          risk10s.append({'id':id, 'object_location':obj['object_location']})

    frame_info.append({'risk3s':risk3s, 'risk5s':risk5s, 'risk10s':risk10s})
    
    datajson[f'frame{i}'] = frame_info

  save_file = open(risk_json_path, "w")  
  json.dump(datajson, save_file)  
    
  print('save file to risk.json')

if __name__ == '__main__':
    # Input
    folderpath = args.folderpath
    videopath = args.videopath
    KML_file_path = args.KML_file_path
    FOV_hor, FOV_ver, CamHeight = args.FOV_horizontal, args.FOV_vertical, args.CameraHeight
    ImageWidth = args.ImageWidth
    ImageHeight = args.ImageHeight


    folder_name = folderpath + '/FullFrame'
    path = Path(folder_name)

    if path.exists() and path.is_dir():
        print(folder_name, 'is exists')
    else:
        path.mkdir()

    print('Set up Finished')
    
    frame_interval = 3
    # Result
    vnp_output_path = folderpath + '/vnp.txt'
    json_file_path = folderpath + '/output.json'
    veclocity_path = folderpath + '/velocity.txt'
    risk_json_path = folderpath + '/risk.json'
    video_path = folderpath + '/video.mp4'

    print('\nVIDEO PATH: ', folderpath)
    print('\nSAVE RESULT TO FOLDER: ', folderpath)
    print('\n====================================================\n')
    print('Save VNP txt to : ', vnp_output_path)
    fps, frame_count = 14, 224
    fps, frame_count = MakeInput(folderpath, vnp_output_path, videopath, frame_interval, stage=2)
    print(fps, frame_count)

    # print('\n====================================================\n')
    # print('KML File process, save velocity to: ', veclocity_path)
    # VelocityExtract(veclocity_path, KML_file_path, fps)
    
    print('\n====================================================\n')
    print('Trajectory, save metadata to: ', json_file_path)
    print('\n====================================================\n')
    ins_matrix_info = [[FOV_hor, FOV_ver], CamHeight]
    TrajectoryAndMakingVideo(videopath, vnp_output_path, veclocity_path, json_file_path, fps, (ImageHeight, ImageWidth), ins_matrix_info, frame_interval) 
    
    risk_json_file = makeJson(frame_count, json_file_path, risk_json_path)
