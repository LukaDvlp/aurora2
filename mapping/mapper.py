#!/usr/bin/env python
""" Mapping module using homography

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30
"""

import math
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color

from aurora.core import core
from aurora.loc import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d as pose
from aurora.loc import transformations as tfm
from aurora.hw import camera


class Mapper():
    def __init__(self, shape, resolution, lamb=1):
        '''
            Init Mapper instance
                size: shape in pixels (height, width)
                resolution: meters in pixel
                lamb: weight to new measurement (0-1)
        '''
        # TODO comment variables
        self.resolution = resolution
        self.shape = shape  # (height, width)
        self.mosaic = np.zeros((self.shape[1], self.shape[0], 3), dtype=np.uint8)
        self.traj = np.zeros(self.mosaic.shape, dtype=np.uint8)
        self.bHi = self.get_homography()
        self.lamb = lamb
        self.center = np.array([s / 2 for s in self.shape[::-1]])  # [u0, v0]
        self.pose = np.append(self.center, [0])  # [u, v, theta]


    def add_image(self, image, wTc):
        '''
            Add new image to map
        '''
        wHb, self.pose = self.compose_homography(wTc)
        map_updated = self.move_map()
        if map_updated:
            wHb, self.pose = self.compose_homography(wTc)
        self.wHi = np.dot(wHb, self.bHi)

        # update map
        new_frame = cv2.warpPerspective(image, self.wHi, self.shape, flags=cv2.INTER_NEAREST)
        self.mosaic[new_frame > 0] += self.lamb * (new_frame[new_frame > 0] - self.mosaic[new_frame > 0])

        # add trajectory point
        cv2.circle(self.traj, tuple(int(x) for x in self.pose[:2]), 1, (0, 0, 255), -1)
        

    
    def get_map(self, trajectory=False, centered=False):
        '''
            Get curremt map
                trajectory: show rover trajectory
                centered: flag for rover-centered map
        '''
        disp = self.mosaic.copy()
        if (trajectory): 
            disp[self.traj > 0] = self.traj[self.traj > 0]
        if (centered):
            R = cv2.getRotationMatrix2D(tuple(self.pose[:2]), -180 / math.pi * self.pose[2], 1)
            R[0, 2] += self.shape[0] / 2 - self.pose[0]
            R[1, 2] += self.shape[0] / 2 - self.pose[1]
            disp = cv2.warpAffine(disp, R, self.shape, flags=cv2.INTER_NEAREST)
        return disp


    def get_rover_view(self, trajectory=False):
        '''
            Get rover-view of the map
        '''
        m = self.get_mosaic(trajectory)
        disp = cv2.warpPerspective(m, np.linalg.inv(self.wHi), self.shape[:2])
        return disp

    
    def get_pose_pix(self, wTc):
        C, pose = self.compose_homography(wTc)
        return pose


    def get_homography(self, dst=(-0.5, 0.6, 1., 1.)):
        '''
            Compute homography between image plane and ground plane
        '''
        dst_yx = self.rect2pts(dst)
        dst_w = np.vstack((dst_yx[::-1,:], np.zeros(4)))
        dst_pix = self.to_pix(dst_yx)
        src_pix = tfm.projectp(dst_w, T=rover.iTb, K=rover.KL)
        return cv2.findHomography(src_pix.T, dst_pix.T)[0]


    def compose_homography(self, wTc):
        p = pose.pose_from_matrix(wTc)
        cy = math.cos(p[2])
        sy = math.sin(p[2])
        C = np.array([[ cy, sy, self.to_pix(p[1]) + self.center[0]],
                      [-sy, cy, self.to_pix(p[0]) + self.center[1]],
                      [  0,  0,  1]])
        pos = np.array([C[0, 2], C[1, 2], p[2]])
        return C, pos


    def rect2pts(self, r):
        p = np.array([[r[0], r[0],      r[0]+r[2], r[0]+r[2]],
                      [r[1], r[1]+r[3], r[1]+r[3], r[1]     ]])
        return p


    def move_map(self):
        '''
            move map so that rover locates within [w/3:2*2/3, h/3:2*h/3]
        '''
        h3, w3 = self.shape[0] / 3, self.shape[1] / 3
        if w3 <= self.pose[0] <= 2 * w3 and h3 <= self.pose[1] <= 2 * h3: return False

        old_mosaic = self.mosaic
        old_traj = self.traj
        self.mosaic = np.zeros(self.mosaic.shape, dtype=np.uint8)
        self.traj = np.zeros(self.mosaic.shape, dtype=np.uint8)
        if self.pose[0] < w3:     
            self.mosaic[:, -2*w3:] = old_mosaic[:, :2*w3]
            self.traj[:, -2*w3:] = old_traj[:, :2*w3]
            self.center[0] += w3
        if self.pose[0] > 2 * w3: 
            self.mosaic[:, :2*w3] = old_mosaic[:, -2*w3:]
            self.traj[:, :2*w3] = old_traj[:, -2*w3:]
            self.center[0] -= w3
        if self.pose[1] < h3:     
            self.mosaic[-2*h3:, :] = old_mosaic[:2*h3, :]
            self.traj[-2*h3:, :] = old_traj[:2*h3, :]
            self.center[1] += h3
        if self.pose[1] > 2 * h3: 
            self.mosaic[:2*h3, :] = old_mosaic[-2*h3:, :]
            self.traj[:2*h3, :] = old_traj[-2*h3:, :]
            self.center[1] -= h3
        return True


    def to_pix(self, meter):
        return meter / self.resolution


## Sample code
if __name__ == '__main__':
    from colorcorrect.algorithm import grey_world
    from aurora.loc import libviso2 as vo

    # load configure from files
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    camera.setup(core.get_full_path('config/camera_local_config.yaml'))
    vo.setup(rover)
    
    # mapper instance
    sz = (500, 500)
    res = 0.02  # m/pix
    vizmap = Mapper(sz, res, lamb=1)

    # rover pose
    wTc = np.eye(4)

    for i in range(1000):
        # load image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None: break
        imL = grey_world(imL)  # correct white balance (optional)
        imR = grey_world(imR)  # correct white balance (optional)
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        # compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        p = pose.update_from_matrix(pTc)
        wTc = pose.matrix_from_pose(p)

        #  mapping
        imLmask = imL
        imLmask[:100, :] = 0
        vizmap.add_image(imLmask, wTc)
        topmap = vizmap.get_map(trajectory=True, centered=True)

        # display
        cv2.imshow('Top view', topmap)
        cv2.waitKey(1)


    plt.pause(1)
    raw_input()  # wait key input

