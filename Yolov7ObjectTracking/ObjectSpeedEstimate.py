from sklearn import svm
import numpy as np 
import argparse
import cv2
import json
import pickle
class EgoCar:
    def __init__(self, ttsfps, look_back=5):
      self.fps = ttsfps
      self.speed = None
      self.x_Distance = [0]
      self.z_Distance = [0]
      self.look_back = look_back
    
    def update_his(self, x_ego, z_ego):
      if (len(self.x_Distance) == self.look_back):
        self.x_Distance = self.x_Distance[1:]
        self.z_Distance = self.z_Distance[1:]
      
      self.x_Distance.append(x_ego+sum(self.x_Distance))
      self.z_Distance.append(z_ego+sum(self.z_Distance))
      
    def TakeAbsoHis(self, ObjHis):
      objectHis = ObjHis
      AbHisX, AbHisZ = [], []      
      for i in range(len(objectHis[0])):
        # print(self.z_Distance[i])
        # print(self.z_Distance[i])
        AbHisX.append(objectHis[0][i] + self.x_Distance[i])
        AbHisZ.append(objectHis[1][i] + self.z_Distance[i])
      return AbHisX, AbHisZ
         
class Object:
    def __init__(self, fps, id, frame_skip, look_back=5):
        self.fps = fps
        self.id = id
        self.speed = None
        self.his = []
        self.step = 0
        self.look_back = look_back
        self.frame_skip = frame_skip

    def TakeHis(self):
      posx, posz = zip(*self.his)
      return [list(posx), list(posz)]    
    
    def update_his(self, location):
      self.step += 1
      if (len(self.his) == self.look_back):
        self.his = self.his[1:]
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
          #khoảng cách từ frame t đến frame 
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

  def LRPredict(self, pos_obj):    
    result = []
    input_data = np.array(pos_obj)
    result = self.lr_model.predict([input_data])
    return result[0]

  def TrajectoryPredict(self, obj_pos, traj_step, predict_step):
      
      arrx, arry = obj_pos[0][-traj_step:], obj_pos[1][-traj_step:]

      PredPosX = self.LRPredict(arrx)
      PredPosZ = self.LRPredict(arry)

      risk = False

      for i in range(len(PredPosX)):  
        if (PredPosX[i] < 4 and PredPosZ[i] < 4): 
          risk = True
      PredPos = [PredPosX, PredPosZ]

      return PredPos, risk

  def PredictRisk(self, frame_id, obj_list, traj_step, predict_step, object_track, EgoCar):
      obj_list = obj_list[1:]
      
      risk_obj_1s = []
      risk_obj_3s = []
      risk_obj_10s = []

      for obj_id in obj_list:
          obj_pos = object_track[obj_id].TakeHis()
          absHis = EgoCar.TakeAbsoHis(obj_pos)
          #Ở đây mình có thể setting thêm một cái tham số appear để quyết định xem có detect nó hay không
          
          predict_step = 1 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(absHis, traj_step, predict_step)
          if (risk == True):
              risk_obj_1s.append(obj_id)

          predict_step = 3 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(absHis, traj_step, predict_step)
          if (risk == True):
              risk_obj_3s.append(obj_id)
          
          predict_step = 10 * int(self.ttfps/self.frame_skip)
          PredPos_, risk = self.TrajectoryPredict(absHis, traj_step, predict_step)
          if (risk == True):
              risk_obj_10s.append(obj_id)

      return risk_obj_1s, risk_obj_3s, risk_obj_10s