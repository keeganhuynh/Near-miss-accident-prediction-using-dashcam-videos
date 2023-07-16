import numpy as np 
import argparse
import cv2
import json
import pickle

class Object:
    def __init__(self, fps, id):
        self.fps = fps
        self.id = id
        self.speed = None
        self.his = []
        self.step = 0
    
    def TakeHis(self, traj_step):
      posx, posz = [], []
      for i in range(len(self.his)):
          posz.append(self.his[i][0])
          posx.append(self.his[i][1])
      pos = [posx, posz]
      return pos

    def update_his(self, location):
      self.step += 1
      if (len(self.his) == int(self.fps)):
        self.his = self.his[1:]
      self.his.append(location)

    def FrameAppear(self):
        return self.step
        
    def speed_predict(self, ego_speed):
        speed = []
        
        count = min(int(self.fps), len(self.his))    
        
        step = 5
        
        if (count <= step):
          return -1
        for i in range(0, count-step, step):
          dis = np.sqrt((self.his[i+step][0]-self.his[i][0])**2 + (self.his[i+step][1]-self.his[i][1])**2)
          
          if (self.his[i+step][0] - self.his[i+step][0] >= 0.0): 
            speed_per_time = dis*(float(self.fps))/float(step)
            speed.append(speed_per_time + ego_speed)

          if (self.his[i+step][0] - self.his[i+step][0] < 0.0):
            speed_per_time = dis*(float(self.fps))/float(step)
            speed.append(speed_per_time - ego_speed)
             
        return float(sum(speed)/len(speed))

    def location_prediction(self):
        return (0,0)

#model_path = '/content/drive/MyDrive/ADAS/.pth'
class Trajectory:
  def __init__(self, model_path, n_steps, n_features):
    self.model = pickle.load(open(model_path, 'rb'))
    self.n_steps = n_steps
    self.n_features = n_features

  def LSMTPredict(self, pos_obj):
    yhat = None
    x = np.array(pos_obj)
    x_input = x.reshape((1, self.n_steps, self.n_features))
    result = self.model.predict(x_input, verbose=0)
    return result[0]

  def TrajectoryPredict(self, obj_pos, traj_step, predict_step, ego_velocity=0, fps=11, ego_turn_angle=0):
      
      arrx, arry, tracX, tracZ = obj_pos[0][-traj_step:], obj_pos[1][-traj_step:], [], []

      UpdataPos = []

      PredPosX = self.LSMTPredict(arrx)
      PredPosZ = self.LSMTPredict(arry)

      risk = False

      for i in range(len(PredPosX)):  
        if (PredPosX[i] < 4 and PredPosZ[i] < 4): 
          risk = True
      
      PredPos = [PredPosX, PredPosZ]

      return PredPos, risk

  def PredictRisk(self, frame_id, obj_list, traj_step, predict_step, object_track, Ego_speed):
      obj_list = obj_list[1:]
      
      risk_obj_3s = []
      risk_obj_5s = []
      risk_obj_10s = []

      for obj_id in obj_list:
          obj_pos = object_track[obj_id].TakeHis(traj_step)
          #Ở đây mình có thể setting thêm một cái tham số appear để quyết định xem có detect nó hay không
          
          predict_step = 33
          PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          if (risk == True):
              risk_obj_3s.append(obj_id)
              


          # predict_step = 55
          # PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          # if (risk == True):
          #     risk_obj_5s.append(obj_id)


          # predict_step = 110
          # PredPos_, risk = self.TrajectoryPredict(obj_pos, traj_step, predict_step)
          # if (risk == True):
          #     risk_obj_10s.append(obj_id)
              
      return risk_obj_3s, risk_obj_5s, risk_obj_10s