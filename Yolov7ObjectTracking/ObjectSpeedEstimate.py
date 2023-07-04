import numpy as np 
import argparse
import cv2
import json

class Object:
    def __init__(self, fps, id):
        self.fps = fps
        self.id = id
        self.speed = None
        self.his = []
        self.step = 1
    
    def TakeHis(self, traj_step):
      posx, posz = [], []
      for i in range(len(self.his)):
          posz.append(self.his[i][0])
          posx.append(self.his[i][1])
      pos = [posx, posz]
      return pos

    def update_his(self, location):
      if (len(self.his) == int(self.fps)):
        self.his = self.his[1:]
        self.step += 1
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

#obj_pos[11] => pos at time stamp
#obj_pos[11] => pos at time stamp
#ego_velocity at time stamp
#input là lịch sử x và z của xe 
#output là dự đoán x và z trong số frame trong tương lai
def TrajectoryPredict(obj_pos, traj_step, predict_step, ego_velocity, fps=11, ego_turn_angle=0):
  arrx, arrz, tracX, tracZ = obj_pos[1], obj_pos[0], [], []

  for i in range(predict_step):
    average = sum(arrx[-traj_step:]) / len(arrx[-traj_step:])
    result = [element for element in arrx[-traj_step:]]
    nexPos = sum(result)/len(result)
    tracX.append(nexPos)
    arrx = np.append(arrx[1:],nexPos)

  for i in range(predict_step):
    average = sum(arrz[-traj_step:]) / len(arrz[-traj_step:])
    result = [element for element in arrz[-traj_step:]]
    nexPos = sum(result)/len(result)
    tracZ.append(nexPos)
    arrz = np.append(arrz[1:],nexPos)

  #return tracX, tracY

  UpdataPos = []

  # PredPosX = [obj_pos[0][-1]]+tracX
  # PredPosZ = [obj_pos[1][-1]]+tracZ
  PredPosX = tracX
  PredPosZ = tracZ

  # TruePosX = obj_pos[0][traj_step:traj_step+predict_step]
  # TruePosZ = obj_pos[1][traj_step:traj_step+predict_step]

  PredPos = [PredPosX, PredPosZ]
  
  risk = False
  
  currposx, currposz = obj_pos[1][-1], obj_pos[0][-1]
  
  
  if (abs(currposx) <= 4 and abs(currposz) <= 5):
      risk = True
      return PredPos, risk
  
  for i in range(predict_step):
    if (abs(PredPosX[i]) <= 4 and abs(PredPosZ[i]) <= 5):
        risk = True

  return PredPos, risk

def PredictRisk(frame_id, obj_list, traj_step, predict_step, object_track, Ego_speed):
    
    obj_list = obj_list[1:]
    Predict_ObjPosList = []
    risk_obj = []
    for obj_id in obj_list:
        obj_pos = object_track[obj_id].TakeHis(traj_step)
        #Ở đây mình có thể setting thêm một cái tham số appear để quyết định xem có detect nó hay không
        PredPos_, risk = TrajectoryPredict(obj_pos, traj_step, predict_step, Ego_speed) #còn cái timestamp chưa fit
        if (risk == True):
            risk_obj.append(obj_id)
    return risk_obj