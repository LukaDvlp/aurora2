#!/usr/bin/env python
""" Mapping module using homography

Input:
    - Left image

Output:
    - Terrain texture image (Forward=v, Left=u)

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-18
"""

import math
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color

import aurora
import imgproc
import localization
import rover


################################################################################
# Mapper
################################################################################

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
        src_pix = imgproc.projectp(dst_w, T=rover.img2base(), K=rover.K())
        return cv2.findHomography(src_pix.T, dst_pix.T)[0]


    def compose_homography(self, wTc):
        xyz = localization.get_position(wTc)
        ang = localization.get_angle(wTc)
        cy = math.cos(ang[0])
        sy = math.sin(ang[0])
        C = np.array([[ cy, sy, self.to_pix(xyz[1]) + self.center[0]],
                      [-sy, cy, self.to_pix(xyz[0]) + self.center[1]],
                      [  0,  0,  1]])
        pose = np.array([C[0, 2], C[1, 2], ang[0]])
        return C, pose


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


################################################################################
# Sample code
################################################################################
if __name__ == '__main__':
    import os
    import glob
    from colorcorrect.algorithm import grey_world

    SRCDIR = '/Users/kyoheee/FieldData/MarsYard2015/freerun03'

    vecT = localization.read_from_file(os.path.join(SRCDIR, 'vo_raw.txt'))
    vecP = localization.concat_motion(vecT, plane_assumption=1)

    resolution = 0.02  # meter/pix
    mapper = Mapper((500, 500), resolution, lamb=1)
    class_mapper = Mapper((500, 500), resolution, lamb=0.6)
    cv2.namedWindow("test")

    files = glob.glob(os.path.join(SRCDIR, 'img/L*.png'))
    filesR = glob.glob(os.path.join(SRCDIR, 'img/R*.png'))
    for i, f in enumerate(files):
        if (i < 1540): continue
        wTc = vecP[i, :].reshape((4, 4))

        img = cv2.imread(f)
        img = grey_world(img)
        imgR = cv2.imread(filesR[i])
        imgR = grey_world(imgR)

        # tilt estimation
        rover.tilt = imgproc.compute_tilt(img, imgR, lamb=0.6)

        # map
        img[:100, :, :] = 0
        mapper.add_image(img, wTc)
        tmap = mapper.get_map(trajectory=True, centered=True)
        tmap = np.fliplr(np.flipud(tmap))
        cv2.imshow("im", tmap)
        cv2.imwrite('/Users/kyoheee/Codes/aurora/gui/static/data/tmap_tmp.png', tmap)
        os.rename('/Users/kyoheee/Codes/aurora/gui/static/data/tmap_tmp.png', '/Users/kyoheee/Codes/aurora/gui/static/data/tmap.png')

        # classification
        label = aurora.classify(img)
        class_mapper.add_image(label, wTc)
        cmap = class_mapper.get_map(centered=True)
        cmap = np.fliplr(np.flipud(cmap))
        cmap_alpha = np.zeros((cmap.shape[0], cmap.shape[1], 4), dtype=np.uint8)
        cmap_alpha[:, :, :3] = cmap
        cmap_alpha[:, :, 3] = 255 * (cmap[:, :, 2] > 0)
        #cv2.imshow("cl", cmap_alpha[:, :, 3])
        cv2.imwrite('/Users/kyoheee/Codes/aurora/gui/static/data/cmap_tmp.png', cmap_alpha)
        os.rename('/Users/kyoheee/Codes/aurora/gui/static/data/cmap_tmp.png', '/Users/kyoheee/Codes/aurora/gui/static/data/cmap.png')

        cv2.waitKey(500)
        #cv2.waitKey(30)

    plt.pause(1)
    raw_input()  # wait key input

