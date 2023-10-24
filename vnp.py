from DeepHough.forward import *
from tqdm import tqdm
import argparse
import cv2
import os

def ProcessVideo(folderpath, videopath, frame_interval=3):
    video_path = videopath
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    nkl_test_file = open('DeepHough/data/training/nkl_test.txt','w')
    nkl_test_file.write('')
    nkl_test_file.close()

    nkl_test_file = open('DeepHough/data/training/nkl_test.txt','w')
    idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=frame_count, desc='Processing frames')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % frame_interval == 0:
            nkl_image_path = f'DeepHough/data/NKL/{idx}.jpg'
            image_path = folderpath + f'/FullFrame/{idx}.jpg'
    
            Cut_Img = frame[int(height/2.5):,:]
    
            cv2.imwrite(nkl_image_path, Cut_Img)
            cv2.imwrite(image_path, frame)
    
            progress_bar.update(1)
            nkl_test_file.write(str(idx)+'\n')
        
        idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print('-----------------------------------')
    print('Frame count: ', frame_count)
    print('Video FPS: ', frame_count)
    print('Total frame proccess: ', int(frame_count/frame_interval))
    print('-----------------+-----------------')
    return int(fps), frame_count

def FirstStage(folderpath, output_path, videopath, stage, frame_interval):
    fps = 0
    frame_count = 0
    if (stage == 0):
      print('===============video processing=================')
      fps, frame_count = ProcessVideo(folderpath, videopath, frame_interval)
    if (stage == 1):
      print('===============Run deep hough=================')
      VanishingPointDetection(output_path, frame_interval)
    if (stage == 2):
      print('===============video processing=================')
      fps, frame_count = ProcessVideo(folderpath, videopath, frame_interval)
      print('===============Run deep hough=================')
      VanishingPointDetection(output_path, frame_interval)
    return fps, frame_count
    

def MakeInput(folderpath, output_path, videopath, frame_interval=3, stage=2):
    return FirstStage(folderpath, output_path, videopath, stage, frame_interval)  

if __name__ == '__main__':
    folderpath = 'Runs/20211213110138_0_8'
    output_path = folderpath + '/vnp.txt'
    videopath = 'Runs/20211213110138_0_8/20211213110138_0_8.avi'
    MakeInput(folderpath, output_path, videopath, 1)
    


