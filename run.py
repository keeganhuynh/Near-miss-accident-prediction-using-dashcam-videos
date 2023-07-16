from setup import SettingEnvironment
SettingEnvironment()
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
args = parser.parse_args()

def ExtractVideo(img_path, frame_count, fps, output_path):
  # Define the output video path
  imgs = []

  for i in range(frame_count):
    try:
      img = cv2.imread(img_path + f'/{i}.jpg')
      imgs.append(img)
    except:
      print(f'No {i}.jpg')


  first_image = (imgs[0])
  frame_height, frame_width, _ = first_image.shape

  # Define the video codec and create a VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  for frame in imgs:    
      out.write(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  out.release()
  cv2.destroyAllWindows()

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


    folder_name = folderpath + '/FullFrame'
    path = Path(folder_name)

    if path.exists() and path.is_dir():
        print(folder_name, 'is exists')
    else:
        path.mkdir()

    print('Set up Finished')
    

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
    fps, frame_count = MakeInput(folderpath, vnp_output_path, videopath, stage=2)
    print(fps, frame_count)

    print('\n====================================================\n')
    print('KML File process, save velocity to: ', veclocity_path)
    VelocityExtract(veclocity_path, KML_file_path, fps)
    
    print('\n====================================================\n')
    print('Trajectory, save metadata to: ', json_file_path)
    print('\n====================================================\n')
    TrajectoryAndMakingVideo(videopath, vnp_output_path, veclocity_path, json_file_path, fps, (1280, 720))
    
    risk_json_file = makeJson(frame_count, json_file_path, risk_json_path)

    ExtractVideo('tam', frame_count, fps, video_path)

