import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

class ObjectClibration:
    def __init__(self, Width, Height, FOV):
        self.Width = Width
        self.Height = Height
        self.alpha = self.Width / (2*np.tan(np.deg2rad(FOV[0]/2)))
        self.beta = self.Height / (2*np.tan(np.deg2rad(FOV[1]/2)))
        self.gamma = 0
        self.intrinsic_matrix = np.array([
                                        [self.alpha, 0, Width//2],
                                        [0, self.beta, Height//2],
                                        [0, 0, 1.0]
                                    ])
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
    
    def get_intrinsic_matrix(self):
        return self.intrinsic_matrix

    def get_extrinsic_matrix(self, vanishing_point):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        
        if vanishing_point is not None:
            p_infinite = np.array([vanishing_point[0], vanishing_point[1], 1]) #correct vanishing point
            inv_intrinsic = np.linalg.inv(self.intrinsic_matrix)

            r3 = np.dot(inv_intrinsic, p_infinite).reshape(-1, )
            r3 = r3/np.linalg.norm(r3)

            alpha = np.arcsin(r3[1])
            beta = -np.arctan(r3[0]/r3[2])
        
        extrinsic_mat = np.array([
            [math.cos(gamma)*math.cos(beta) + math.sin(alpha)*math.sin(beta)*math.sin(gamma), math.cos(gamma)*math.sin(alpha)*math.sin(beta) - math.cos(beta)*math.sin(gamma), - math.cos(alpha)*math.sin(beta)],
            [math.cos(alpha)*math.sin(gamma), math.cos(alpha)*math.cos(gamma), math.sin(alpha)],
            [math.cos(gamma)*math.sin(beta) - math.cos(beta)*math.sin(alpha)*math.sin(gamma), -math.cos(gamma)*math.cos(beta)*math.sin(alpha)-math.sin(gamma)*math.sin(beta), math.cos(alpha)*math.cos(beta)],
        ])
        return extrinsic_mat

def uv_to_world(object_coor, cam_height, vnp, image_shape, FOV):
    
    object_ = ObjectClibration(image_shape[1], image_shape[0],FOV)

    extrinsic_mat = object_.get_extrinsic_matrix(vnp)
    intrinsic_matrix = object_.intrinsic_matrix

    point = object_coor[0], object_coor[1]
    point = np.concatenate((point, np.array([1]))).reshape(-1, 1)
    
    n_c = extrinsic_mat @ np.array([0, 1, 0]).reshape(-1, 1)
    
    inv_extrinsic_mat = np.linalg.inv(extrinsic_mat)
    inv_intrinsic_mat = np.linalg.inv(intrinsic_matrix)
    
    lambd = cam_height / float((n_c.T @  inv_intrinsic_mat @ point).squeeze())
    
    cam_coords = lambd * inv_intrinsic_mat @ point
    
    world_coos = inv_extrinsic_mat @ cam_coords
    
    return world_coos.squeeze()

def CalSafeDistance(velocity, surface_type = 0.85, g = 9.8):
    return velocity*velocity / (2*surface_type*g)