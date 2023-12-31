import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
import random
import time

def VisuallizeMV(P0, Pk, image):
    print(len(P0))
    print(len(Pk))
    for i in range(len(P0)):
        x = [P0[i][0], Pk[i][0]]
        y = [P0[i][1], Pk[i][1]]
        plt.plot(x, y, color='red', linewidth=2)

    plt.imshow(image)
    plt.show()
    
def VisuallizeIntersectPoint(points, image):
    for point in points:
        plt.scatter(point[0], point[1], c='red', s=5)
    
    plt.imshow(image)
    plt.show()

def VideoRead(video_path, skip_frame, resize_factor=None):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    n_frame = 0
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        print("Không thể mở video.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if n_frame % skip_frame == 0:
                if resize_factor is not None:
                    # Thu nhỏ kích thước khung hình nếu có yêu cầu
                    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
                frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            n_frame += 1

        cap.release()
        
    return frame_list, fps, frame_count

class MVextractor:
    def __init__(self, video_path, skip_frame, resize_factor=1, nt0=500, TD=0, TN=450, l=1, RANSAC_iter=45):
        self.RANSAC_iter = RANSAC_iter
        self.video_path = video_path
        self.skip_frame = skip_frame
        self.nt0 = nt0
        self.TD = TD
        self.TN = TN
        self.l = l
        self.frame_list, self.fps, self.frame_count = VideoRead(video_path, skip_frame, resize_factor)
        self.height = self.frame_list[0].shape[0]
        self.width = self.frame_list[0].shape[1] 
        # print(self.height, self.width)
        return

    def mv_detection(self, initial_fr, k, P0_corner, P0k_corner):
        frame_list = self.frame_list
        TD = self.TD

        if (len(P0k_corner) == 0):
            P_previous_k = P0_corner
        else:
            P_previous_k = P0k_corner
        
        Pk = []

        succ_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_list[initial_fr], frame_list[initial_fr+k], P0_corner, None)
        
        tracked_status = status.flatten()
        succ_points = succ_points.reshape(-1,2)
        
        indices_to_remove = np.where(tracked_status==0)
        Pk = succ_points[tracked_status==1] 
        
        P0_corner = np.delete(P0_corner, indices_to_remove, axis=0)
        P_previous_k = np.delete(P_previous_k, indices_to_remove, axis=0)

        # VisuallizeMV(P0_corner, succ_points[tracked_status==1], frame_list[initial_fr])

        Pk = np.array(Pk)

        Pk_temporary = Pk
        P_previous_k_temporary = P_previous_k

        indices_to_remove = []
        
        for idx in range(len(Pk_temporary)):
            x1, y1, x2, y2 =  Pk_temporary[idx][0], Pk_temporary[idx][1], P_previous_k_temporary[idx][0], P_previous_k_temporary[idx][1]
        
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
            if (distance < TD):   
                indices_to_remove.append(idx)
        
        P0_corner = np.delete(P0_corner, np.where(indices_to_remove), axis=0)
        Pk = np.delete(Pk, np.where(indices_to_remove), axis=0)

        return np.array(P0_corner), np.array(Pk)

    def extended_point(self, p0, pk):
        l = self.l #extend_pixel_thres = self.l
        x1, y1, x2, y2 = p0[0], p0[1], pk[0], pk[1]
        
        x = x2 - x1
        y = y2 - y1
        l2 = math.sqrt((x2-x1)**2+(y2-y1)**2)

        if (l2 == 0):
            return x, y
        x2 += l/l2 * x
        y2 += l/l2 * y

        return x2, y2
    
    def mv_selection(self, P0, Pk):
        height, width = self.height, self.width
        pc_x, pc_y = width, height
        delete_mv_type2 = []

        for i in range(len(Pk)):
            x1,y1 = P0[i][0], P0[i][1]
            x2,y2 = self.extended_point(P0[i], Pk[i])
            d_basepoint = np.sqrt((x1-pc_x)**2+(y1-pc_y)**2)
            d_headpoint = np.sqrt((x2-pc_x)**2+(y2-pc_y)**2)
            if d_headpoint < d_basepoint:
                delete_mv_type2.append(i)
        
        rawP0, rawPk = P0, Pk
        P0 = np.delete(rawP0, delete_mv_type2, axis=0)
        Pk = np.delete(rawPk, delete_mv_type2, axis=0)

        delete_mv_type3 = []
        for i in range(len(Pk)):
            vector = Pk[i] - P0[i]
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            if angle > -10 and angle < 10:
                delete_mv_type3.append(i)
        
        rawP0, rawPk = P0, Pk
        P0 = np.delete(rawP0, delete_mv_type3, axis=0)
        Pk = np.delete(rawPk, delete_mv_type3, axis=0)

        return P0, Pk
    
    def AngleCal(self, A, B):

        if len(A) != len(B):
            raise ValueError('''The lengths of A and B must be the same''')

        angles = np.zeros(len(A))
        
        for i in range(len(A)):
            vector1 = A[i]
            vector2 = B[i]

            norm_A = np.sqrt(int(vector1[0]*vector1[0] + vector1[1]*vector1[1]))
            norm_B = np.sqrt(int(vector2[0]*vector2[0] + vector2[1]*vector2[1]))

            # t0 = time.time()
            dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1] #MIN
            # t1 = time.time()
            # dot_calculation.append(t1-t0)
            
            if dot_product < -norm_A * norm_B:
                angles[i] = 180.0
            elif dot_product > norm_A * norm_B:
                angles[i] = 0.0
            else:
                theta_rad = np.arccos(dot_product / (norm_A * norm_B))
                theta_degrees = np.degrees(theta_rad)
                angles[i] = theta_degrees    

        return angles
    
    def IntersectPoint(self, p1, p2, q1, q2):
        height, width = self.height, self.width
        vector_A = p2 - p1
        vector_B = q2 - q1
        slope_A = None
        slope_B = None

        if vector_A[0] != 0:
            slope_A = vector_A[1] / vector_A[0]

        if vector_B[0] != 0:
            slope_B = vector_B[1] / vector_B[0]

        if slope_A is not None and slope_B is not None:
            c = p1[1] - slope_A * p1[0]
            d = q1[1] - slope_B * q1[0]

            if slope_A != slope_B:
                x_intersect = (d - c) / (slope_A - slope_B)
                y_intersect = slope_A * x_intersect + c
                intersection_point = np.array([int(x_intersect), int(y_intersect)])
                x, y = intersection_point[0], intersection_point[1]
                if x < width and x > 0 and y < height and y > 0:
                    return intersection_point, True
                else:
                    return None, False
            else:
                return None, False
        else:
            # Xử lý đoạn thẳng dọc
            # Ở đây bạn có thể thêm mã xử lý khi cả hai đoạn thẳng là đoạn thẳng dọc
            return None, False
    
    def ScoreRS(self, angles):
        results = []

        for angle in angles:
            if abs(angle) > 45:
                continue
            else:
                results.append(math.exp(-1*abs(angle)))

        return sum(results)

    def Angle_RANSAC_voting(self, P0, Pk, iter):
        best_score = -1

        RVP = None
        Hypothesis_RVP = None
        chosen_point = []
        
        for _ in range(iter):
            flag = False
            #random_vector ~ rv
            while flag == False:       
                rv1 = random.randint(0, len(P0)-1)
                rv2 = random.randint(0, len(P0)-1) 
                Hypothesis_RVP, flag = self.IntersectPoint(P0[rv1], Pk[rv1], P0[rv2], Pk[rv2]) 
            
            vector_A, vector_B = [], []
            
            for i in range(len(Pk)):
                vector_A.append(Pk[i] - P0[i])
                vector_B.append(Pk[i] - Hypothesis_RVP)
            
            angles = self.AngleCal(vector_A, vector_B)
            
            score = self.ScoreRS(angles)
            
            if score > best_score:
                best_score = score
                RVP = Hypothesis_RVP    
        
        return RVP, chosen_point
    
    def SaveTxt(self, txt, save_path):
        with open(save_path, 'w') as txt_file:
            txt_file.write(txt)

    def R_VP_detection(self, save_path, initial_frame=0):
        txt = ""
        result = []
        nt0, TD, TN, l, RANSAC_iter = self.nt0, self.TD, self.TN, self.l, self.RANSAC_iter
        frame_list = self.frame_list

        P_0, P_k = None, []
        frame_count = len(frame_list)
        
        k = 1
        P0_corner = cv2.goodFeaturesToTrack(frame_list[initial_frame], maxCorners=nt0, qualityLevel=0.01, minDistance=10).reshape(-1,2)
        P0k_corner = []
        
        while (True):    
            while (True):
                P_0, P_k = self.mv_detection(initial_frame, k, P0_corner, P0k_corner)
                        
                V_tk = len(P_0)
                
                if V_tk > TN:
                    break
                else:
                    k = 1
                    initial_frame = initial_frame + k

                    if initial_frame >= frame_count - 1:
                        self.SaveTxt(txt, save_path)
                        return
                    
                    P0_corner = cv2.goodFeaturesToTrack(frame_list[initial_frame], maxCorners=nt0, qualityLevel=0.01, minDistance=10).reshape(-1,2)
                    P0k_corner = []

            P_0, P_k = self.mv_selection(P_0, P_k)
            # VisuallizeMV(P_0, P_k, frame_list[initial_frame])
            
            t0 = time.time()
            
            RVP, rvnplist = self.Angle_RANSAC_voting(P_0, P_k, iter=self.RANSAC_iter)
            result.append(RVP)
            
            for _ in range(self.skip_frame):
                txt += f'{self.skip_frame*initial_frame+_},{RVP[0]},{RVP[1]}\n'
                
            # print(f'{initial_frame}/{len(frame_list)} : {RVP}')
            # VisuallizeIntersectPoint(rvnplist, frame_list[initial_frame])
            
            # print(f'------------------------- {initial_frame} -----------------------------')
            # plt.scatter(RVP[0], RVP[1], c='red', s=50, zorder=100)
            # plt.imshow(frame_list[initial_frame])
            # plt.show()
            # plt.close()
            
            # print(f'frame {initial_frame} : \n{RVP}\n')

            k += 1
            P0_corner = P_0
            P0k_corner = P_k
            if initial_frame + k >= frame_count - 1:
                self.SaveTxt(txt, save_path)
                return
            
def organize_data(data, frame_count):
    organized_data = {}

    for line in data:
        index, x, y = map(int, line.split(","))
        organized_data[index] = (x, y)

    result = []
    for i in range(frame_count):
        if i in organized_data:
            result.append((i, *organized_data[i]))
        else:
            closest_index = min(organized_data.keys(), key=lambda k: abs(k - i))
            result.append((i, *organized_data[closest_index]))

    return result

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def write_result_to_file(result, output_file_path):
    with open(output_file_path, 'w') as file:
        for line in result:
            file.write(','.join(map(str, line[1:])) + '\n')