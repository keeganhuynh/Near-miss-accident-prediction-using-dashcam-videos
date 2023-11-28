from sklearn import svm
import numpy as np 
import argparse
import cv2
import json
import pickle

class Object:
    def __init__(self, fps, id, frame_skip=1):
        self.fps = fps
        self.id = id
        self.speed = None
        self.his = []
        self.absolote_his = []
        self.step = 0
        self.frame_skip = frame_skip
    
    def TakeHis(self):
      posx, posz = zip(*self.his)
      return [list(posx), list(posz)]   
    
    def TakeAbsoHis(self):
      posx, posz = zip(*self.absolote_his)
      return [list(posx), list(posz)]  
    
    def update_his(self, location):
      self.step += 1
      if (len(self.his) == int(self.fps/self.frame_skip)):
        self.his = self.his[1:]
        self.absolote_his = self.absolote_his[1:]
      
      #z = coors[2]-ego_car[2], x = coors[0]-ego_car[0]
      #z = location[0], x = location[1]

      self.his.append(location)

    def FrameAppear(self):
        return self.step
        
    def speed_predict(self, ego_speed):
        speed = []
        
        count = min(int(self.fps/self.frame_skip), len(self.his))    
        
        step = 5
        
        if (count <= step):
          return -1
        for i in range(0, count-step, step):
          dis = np.sqrt((self.his[i+step][0]-self.his[i][0])**2 + (self.his[i+step][1]-self.his[i][1])**2)
          
          if (self.his[i+step][0] - self.his[i+step][0] >= 0.0): 
            speed_per_time = dis*(float(self.fps/self.frame_skip))/float(step)
            speed.append(speed_per_time + ego_speed)

          if (self.his[i+step][0] - self.his[i+step][0] < 0.0):
            speed_per_time = dis*(float(self.fps/self.frame_skip))/float(step)
            speed.append(speed_per_time - ego_speed)
             
        return float(sum(speed)/len(speed))

    def location_prediction(self):
        return (0,0)

#model_path = '/content/drive/MyDrive/ADAS/.pth'
#Yolov7ObjectTracking
#0 : SVM : svm_model.pkl
#1 : LSTM : lstm11s.pth
#2 : Linear Regression : LR_model.pkl
class Trajectory:
  def __init__(self, n_steps=5, n_features=1, model_options=2, ttfps=15, frame_skip=1):
    if model_options == 2:
      model_path = 'Yolov7ObjectTracking/LR_model.pkl'
      self.lr_model = pickle.load(open(model_path, 'rb'))

    self.n_steps = n_steps
    self.n_features = n_features
    self.ttfps = ttfps
    self.frame_skip = frame_skip
  def absoluteHis(self):
     
      return 
  def LRPredict(self, pos_obj):    
    result = []
    
    input_data = np.array(pos_obj)
    result = self.lr_model.predict([input_data])
       
    return result[0]

  def TrajectoryPredict(self, obj_pos, traj_step, predict_step, ego_velocity=0, fps=11, ego_turn_angle=0):
      
      arrx, arry = obj_pos[0][-traj_step:], obj_pos[1][-traj_step:]

      PredPosX = self.LRPredict(arrx)
      PredPosZ = self.LRPredict(arry)

      risk = False

      for i in range(len(PredPosX)):  
        if (PredPosX[i] < 4 and PredPosZ[i] < 4): 
          risk = True
      
      PredPos = [PredPosX, PredPosZ]

      return PredPos, risk

  def PredictRisk(self, frame_id, obj_list, traj_step, predict_step, object_track, Ego_speed, ego_car):
      obj_list = obj_list[1:]
      
      risk_obj_1s = []
      risk_obj_3s = []
      risk_obj_10s = []

      for obj_id in obj_list:
          obj_pos = object_track[obj_id].TakeHis()
          #Ở đây mình có thể setting thêm một cái tham số appear để quyết định xem có detect nó hay không
          
          predict_step = 1 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          if (risk == True):
              risk_obj_1s.append(obj_id)

          predict_step = 3 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          if (risk == True):
              risk_obj_3s.append(obj_id)
          
          predict_step = 10 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          if (risk == True):
              risk_obj_10s.append(obj_id)


      return risk_obj_1s, risk_obj_3s, risk_obj_10s