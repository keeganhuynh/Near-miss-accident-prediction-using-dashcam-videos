from setup import SettingEnvironment
SettingEnvironment()

from vnp import *
from Yolov7ObjectTracking.detect_and_track import *
from SpeedETool import *
import cv2

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--folderpath', help='')
parser.add_argument('--videopath', help='')
parser.add_argument('--KML_file_path', help='')
args = parser.parse_args()

def ExtractVideo(img_path, frame_count, fps):
  # Define the output video path
  output_path = 'output_video.mp4'
  imgs = []

  for i in range(frame_count):
    img = cv2.imread(img_path + f'/{i}.jpg')
    imgs.append(img)


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

if __name__ == '__main__':
    
    print('Set up Finished')

    # Input
    folderpath = args.folderpath
    videopath = args.videopath
    KML_file_path = args.KML_file_path
    

    # Result
    vnp_output_path = folderpath + '/vnp.txt'
    json_file_path = folderpath + '/output.json'
    veclocity_path = folderpath + '/velocity.txt'

    print('\nVIDEO PATH: ', folderpath)
    print('\nSAVE RESULT TO FOLDER: ', folderpath)
    print('\n====================================================\n')
    print('Save VNP txt to : ', vnp_output_path)
    fps, frame_count = MakeInput(folderpath, vnp_output_path, videopath, stage=2)

    print('\n====================================================\n')
    print('KML File process, save velocity to: ', veclocity_path)
    VelocityExtract(veclocity_path, KML_file_path,fps)
    
    print('\n====================================================\n')
    print('Trajectory, save metadata to: ', json_file_path)
    print('\n====================================================\n')
    TrajectoryAndMakingVideo(videopath, vnp_output_path, veclocity_path, json_file_path, fps)

    ExtractVideo('Frame_Extract', frame_count, fps)
    print('Save video')
    

