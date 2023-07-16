import numpy as np
import matplotlib.pyplot as plt


def TakeHistory(idx, frame_id, traj_step, predict_step):
    posx = []
    posz = []
    
    frame = f'frame{frame_id}'
    egocarsped = data['FrameInfo'][0][frame][0]['velocity']

    if (frame_id < traj_step):
      return [], egocarsped

    for i in range(frame_id-traj_step, frame_id+predict_step+1):
        frame = f'frame{i}'
        for point in data['FrameInfo'][0][frame]:
            if (point['id'] == idx):
                x = point['object_location'][0]['X']
                z = point['object_location'][0]['Z']
                posx.append(x)
                posz.append(z)
    
    if (len(posx) < traj_step):
      return [], egocarsped
    if (len(posx) < traj_step):
      return [], egocarsped

    pos = [posx, posz]

    return pos, egocarsped
#obj_pos[11] => pos at time stamp
#obj_pos[11] => pos at time stamp
#ego_velocity at time stamp

#input là lịch sử x và z của xe 
#output là dự đoán x và z trong số frame trong tương lai
def TrajectoryPredict(obj_pos, traj_step, predict_step, ego_velocity, fps=11, ego_turn_angle=0):
    arrx, arry, tracX, tracZ = obj_pos[0][0:traj_step], obj_pos[1][:traj_step], [], []
    
    #step is range predict
    for i in range(predict_step):
      average = sum(arrx[-6:]) / len(arrx[-6:])
      result = [element for element in arrx[-6:] if (element <= average * 1)]
      nexPos = sum(result)/len(result)
      tracX.append(nexPos)
      arrx = np.append(arrx[1:],nexPos)

    for i in range(predict_step):
      average = sum(arry[-6:]) / len(arry[-6:])
      result = [element for element in arry[-6:] if (element <= average * 1)]
      nexPos = sum(result)/len(result)
      tracZ.append(nexPos)
      arry = np.append(arry[1:],nexPos)

    #return tracX, tracY

    UpdataPos = []

    PredPosX = [obj_pos[0][traj_step]]+tracX
    PredPosZ = [obj_pos[1][traj_step]]+tracZ
    TruePosX = obj_pos[0][traj_step:traj_step+predict_step]
    TruePosZ = obj_pos[1][traj_step:traj_step+predict_step]

    for i in range(1, predict_step):
      PredPosZ[i] += ego_velocity*(i+1)/(11*3.6)
      
      #Trường hợp ground truth không còn thì ko detect chi nữa
      try:
        TruePosZ[i] += ego_velocity*(i+1)/(11*3.6)
      except:
        TruePosZ.append(PredPosZ[i])
    
    TruePos = [TruePosX, TruePosZ]
    PredPos = [PredPosX, PredPosZ]
    
    risk = False

    for i in range(len(TruePos)):  
      NextPosZ = ego_velocity*((i+1)/fps)
      if (PredPosX[i] <= 1.5):
        if (abs(PredPosZ[i] - NextPosZ) <= 1.5):
          risk = True

    return PredPos, TruePos, risk

def PredictRisk(frame_id, obj_list, traj_step, predict_step):
    
    obj_list = obj_list[1:]
    Predict_ObjPosList = []
    Groundtruth_ObjPosList = []
    
    Ego_speed = -1
    risk_obj = []

    for obj_id in obj_list:
        obj_pos, Ego_speed = TakeHistory(obj_id, frame_id, traj_step, predict_step)

        if (len(obj_pos) == 0 or len(obj_pos[0]) != (traj_step + predict_step + 1)):
            continue
        
        #Ở đây mình có thể setting thêm một cái tham số appear để quyết định xem có detect nó hay không
        TruePos_, PredPos_, risk = TrajectoryPredict(obj_pos, traj_step, predict_step, Ego_speed) #còn cái timestamp chưa fit
        
        if (risk == True):
            risk_obj.append(obj_id)
        
        Groundtruth_ObjPosList.append(TruePos_)
        Predict_ObjPosList.append(PredPos_)

    return risk_obj
    # DrawMapWithPos(Ego_speed, Groundtruth_ObjPosList, Predict_ObjPosList, traj_step, predict_step)
  

def RGB2RGBA(col):
    color = col[0]/255, col[1]/255, col[2]/255
    return color

def DrawEachObj(TruePos, PredPos, realsize, size, step):
    ogrp = ((TruePos[0][0]+8)/realsize*size, TruePos[1][0]/realsize*2*size)
    circle2 = plt.Circle(ogrp, 40, edgecolor='black', color = RGB2RGBA((0, 128, 255)))
    circle2.set_zorder(3)

    scatX =  [int((i+8)/realsize*size) for i in TruePos[0][:step]]
    scatZ = [int(i/realsize*2*size) for i in TruePos[1][:step]]
    scatX = [i for i in scatX if (i>0 and i<600)]
    numX = len(scatX)
    scatZ = [i for i in scatZ if (i>0 and i<1200)]
    numZ = len(scatZ)
    gt_step = numZ
    if (numX > numZ):
      gt_step = numZ
      scatX = scatX[:numZ]
    if (numX < numZ):
      gt_step = numX
      scatZ = scatZ[:numX]

    pred_scatX =  [int((i+8)/realsize*size) for i in PredPos[0][:step]]
    pred_scatZ = [int(i/realsize*2*size) for i in PredPos[1][:step]]

    pred_scatX = [i for i in pred_scatX if (i>0 and i<600)]
    numX = len(pred_scatX)
    pred_scatZ = [i for i in pred_scatZ if (i>0 and i<1200)]
    numZ = len(pred_scatZ)
    pre_step = numZ

    if (numX > numZ):
      pre_step = numZ
      pred_scatX = pred_scatX[:numZ]
    if (numX < numZ):
      pre_step = numX
      pred_scatZ = pred_scatZ[:numX]

    for id in range(pre_step-1):
      plt.plot([pred_scatX[id], pred_scatX[id+1]], [pred_scatZ[id], pred_scatZ[id+1]], color = 'green')

    for id in range(gt_step-1):
      plt.plot([scatX[id], scatX[id+1]], [scatZ[id], scatZ[id+1]], color = 'blue')


    plt.scatter(pred_scatX, pred_scatZ, color = 'red')
    plt.scatter(scatX, scatZ)

    return plt, circle2

def DrawMapWithPos(egocar_speed, TruePosList, PredPosList, traj_step, predict_step, fps=11, number_of_object=0):
    # Tạo đối tượng subplot
    if (len(TruePosList) <= 0 or len(PredPosList) <=0):
      return

    fig, ax = plt.subplots(figsize=(5,10))
    size = 600
    realsize = 16
    plt.plot([0, 0], [0, 2*size], color='black')
    plt.plot([0, size], [0, 0], color='black')
    plt.plot([0, size], [2*size, 2*size], color='black')
    plt.plot([size, size], [2*size, 0], color='black')

    plt.plot([6.5/realsize*size, 6.5/realsize*size], [0, 2*size], color='black', linewidth = 0.5)
    plt.plot([9.5/realsize*size, 9.5/realsize*size], [0, 2*size], color='black', linewidth = 0.5)

    text = "EgoCar"
    
    #FPS here
    for i in range(predict_step):
      ego_nexpos = int((egocar_speed*((i+1)/fps))/(3.6*realsize)*2*size)
      
      if (ego_nexpos >= 1200):
        ego_nexpos = 1200

      plt.plot([size//2, size//2], [0, ego_nexpos], color = 'green', \
              marker='o', linestyle='dashed', linewidth=5, markersize=12)

    ogrp = (size/2, 0)
    ax.text(ogrp[0], ogrp[1], text, fontsize=8, color='black', ha='center', va='center')
    ax.text(100, 15, f'Egocar speed: {egocar_speed}', fontsize=8, color='black', ha='center', va='center')


    circle1 = plt.Circle(ogrp, 40, edgecolor='black', color = RGB2RGBA((221, 161, 94)))
    ax.add_patch(circle1)
    circle1.set_zorder(2)

    #CHECK OBJ
    for i in range(len(TruePosList)):
      TruePos = TruePosList[i]
      PredPos = PredPosList[i]

      ogrp = ((TruePos[0][0]+8)/realsize*size, TruePos[1][0]/realsize*2*size)

      if (ogrp[0] < 0 or ogrp[0]>600):
        continue
      if (ogrp[1] < 0 or ogrp[1]>1200):
        continue

      _, cir = DrawEachObj(TruePos, PredPos, realsize, size, traj_step)
      ax.add_patch(cir)

    plt.show()
    plt.close()