from DeepHough.forward import *
from tqdm import tqdm
import argparse
import cv2
import os

def ProcessVideo(folderpath, videopath):
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
    print('Frame count: ', frame_count)
    return int(fps), frame_count

def FirstStage(folderpath, output_path, videopath, stage=2):
    fps = 0
    frame_count = 0
    if (stage == 0):
      print('===============video processing=================')
      fps, frame_count = ProcessVideo(folderpath, videopath)
    if (stage == 1):
      print('===============Run deep hough=================')
      VanishingPointDetection(output_path)
    if (stage == 2):
      print('===============video processing=================')
      fps, frame_count = ProcessVideo(folderpath, videopath)
      print('===============Run deep hough=================')
      VanishingPointDetection(output_path)
    return fps, frame_count
    

def MakeInput(folderpath, output_path, videopath, stage=2):
    # folderpath = 'Runs/20211213110138_0_8'
    # output_path = folderpath + '/vnp.txt'
    return FirstStage(folderpath, output_path, videopath, stage)  

if __name__ == '__main__':
    folderpath = 'Runs/20211213110138_0_8'
    output_path = folderpath + '/vnp.txt'
    videopath = 'Runs/20211213110138_0_8/20211213110138_0_8.avi'
    MakeInput(folderpath, output_path, videopath, 1)
    


