import numpy as np
import time

from collections import deque

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import cv2

class Line2D:
    def __init__(self, a, b, c, properties={}) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.properties = properties
    def add_properties(self, extra_prop):
        self.properties = {**self.properties, **extra_prop}


def get_line_from_points(point1, point2):
    direction_vec = point1-point2
    if direction_vec[0] <= 0:
        direction_vec *= -1
    n_vec = np.array([-direction_vec[1], direction_vec[0]])
    x_min = min(point1[0], point2[0])
    return Line2D(n_vec[0], n_vec[1], -(n_vec[0]*point1[0]+n_vec[1]*point1[1]), {"x_min": x_min})


def calculate_intersect(l1: Line2D, l2: Line2D):
    if l1.a / l2.a == l1.b / l2.b and l1.b / l2.b == l1.c / l2.c:
        return "MUTLI", l1
    if l1.a / l2.a == l1.b / l2.b:
        return "NONE", None
    mat = np.array([[l1.a, l1.b], [l2.a, l2.b]])
    vec = -1 * np.array([l1.c, l2.c]).reshape(-1, 1)
    
    return "ONE", (np.linalg.inv(mat)@vec).squeeze()


def linear_regr_approx(points):
    x_min = np.min(points, axis=0)[0]
    linear_model = LinearRegression(fit_intercept=False)
    points = points.T
    x_train = np.concatenate((np.ones((points.shape[1], 1)), points[0].reshape(-1, 1)), axis=1)
    y_train = points[1].reshape(-1, 1)
    linear_model.fit(x_train, y_train)
    score = linear_model.score(x_train, y_train)
    pred = linear_model.predict(x_train)
    loss = mean_squared_error(y_train, pred)
    coef = linear_model.coef_.squeeze()
    return Line2D(-coef[1], +1, -coef[0], {"x_min": x_min, "loss": round(loss, 2), "score": round(score, 2)})

def calculate_angle(l1, l2):
    cos_value = (l1.a*l2.a + l1.b*l2.b) / (np.linalg.norm(np.array([l1.a, l1.b])) * np.linalg.norm(np.array([l2.a, l2.b])))
    angle_rad = np.arccos(cos_value)
    return angle_rad

class TurnDetector:
    MIN_NUM_FEAT = 2000
    def __init__(self, intrinsic_matrix):
        self.intrinsic_mat = intrinsic_matrix
        self.prev_status = 0
        self.list_status = deque(maxlen = 5)
        self.sift = cv2.SIFT_create()
        self.prev_frame = None
        self.prev_feats = None
        self.prev_angle = None

    def process(self, current_frame, motion_vectors):
        angle = 0
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return 0

        turn_stable = self.isTurn(motion_vectors)

        if turn_stable != 0:

            if self.prev_feats is None or len(self.prev_feats) < self.MIN_NUM_FEAT:
                self.prev_feats = self.featureDetection(self.prev_frame)

            prev_features, curr_feature = self.featureTracking(self.prev_frame, current_frame, self.prev_feats, turn_stable)
            E, mask_ = cv2.findEssentialMat(prev_features,curr_feature,self.intrinsic_mat,method=cv2.RANSAC,prob=0.999,threshold=1.0)
            points, rmat, tvec, mask_ = cv2.recoverPose(E, prev_features, curr_feature)

            (y, p, r), _ = cv2.Rodrigues(rmat) 

            angle = np.abs(p[0])
            # angle = p[0]
            # if self.prev_angle is not None and angle*self.prev_angle < 0:
            #     angle = angle / np.abs(angle) * np.pi/180
            if turn_stable == -1:
                angle *= -1
        

            self.prev_feats = prev_features.copy()
        self.prev_frame = current_frame.copy()
        # self.total_angle += angle
        # if angle != 0: print(f"Turning {np.rad2deg(angle)} degrees")
        return angle #radian


    def featureDetection(self, img_1):
        keypoints_1, des = self.sift.detectAndCompute(img_1, None)

        points1 = cv2.KeyPoint_convert(keypoints_1)
        return points1

    def featureTracking(self, img_1, img_2, points1, turn):
        points2, status, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, points1, None, winSize=(21,21), maxLevel=3, 
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        index_correction = 0
        for i in range(len(status)):
            p1 = points1[i - index_correction]
            p2 = points2[i - index_correction]

            length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            
            if (status[i] == 0) or (p2[0] < 0) or (p2[1] < 0) or length > 30 or length < 5 or (turn*(p2[0] - p1[0]) > 0):
                if (p2[0] < 0) or (p2[1] < 0) or length > 30:
                    status[i] = 0

                p1 = points1[i-index_correction]
                p2 = points2[i-index_correction]
                points1 = np.delete(points1, i-index_correction, axis=0)
                points2 = np.delete(points2, i-index_correction, axis=0)
                index_correction += 1

        return points1, points2

    def isTurn(self, motion_vectors):
        """
        _summary_

        Args:
            motion_vectors (_type_): _description_

        Returns:
            -1: left
            0: no turn
            1: turn right
        """        
        turn_stable = self.prev_status

        if len(motion_vectors) > 0:
            num_bin = 10
            angle_bin = np.zeros(num_bin)
            leff_right_bin = np.zeros((num_bin, 2))

            # leff_right_bin =  [[0,0],[0,0], ..., [0,0]]

            num_mvs = np.shape(motion_vectors)[0]

            for mv in np.split(motion_vectors, num_mvs):
                start_pt = np.array((mv[0, 3], mv[0, 4]))
                end_pt = np.array((mv[0, 5], mv[0, 6]))
                
                length = np.sqrt((start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2)

                if length > 50 or length < 5:
                    continue

                angle = np.rad2deg(calculate_angle(get_line_from_points(start_pt, end_pt), get_line_from_points(np.array([0, 0]), np.array([10, 0]))))
                leff_right = 0 if end_pt[0] < start_pt[0] else 1
                id = int(angle // (180/num_bin))
                leff_right_bin[id, leff_right] += 1
                angle_bin[id] += 1

            most_bin = np.argmax(angle_bin)

            if angle_bin[most_bin] > 1000:
                curr_status = np.argmax(leff_right_bin[most_bin])
                if curr_status == 1:
                    curr_status = -1
                elif curr_status == 0:
                    curr_status = 1
            else:
                curr_status = 0
                
            self.list_status.append(curr_status)

            if np.count_nonzero(np.array(self.list_status)==0) > 2:
                turn_stable = 0
            else:
                if curr_status != 0:
                    turn_stable = curr_status
                else:
                    turn_stable = self.prev_status
        self.prev_status = turn_stable
        return turn_stable