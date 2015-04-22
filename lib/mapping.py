#!/usr/bin/env python
""" Mapping module using homography

Input:
    - Left image

Output:
    - Terrain texture image

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-18
"""

import math
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2

import aurora


class Mapper():
    def __init__(self, size=(800, 800), resolution=0.025, lamb=1):
        self.resolution = resolution
        self.size = size
        self.mosaic = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.traj = np.zeros(self.mosaic.shape, dtype=np.uint8)
        self.bHi = self.get_homography()
        self.lamb = lamb
        pass


    def add_image(self, image, wTc):
        xyz = aurora.localization.get_position(wTc)
        ang = aurora.localization.get_angle(wTc)
        wHb = self.compose_homography(xyz[0], xyz[1], ang[0])
        #wHb = self.compose_homography(0, 0, math.pi)
        self.wHi = np.dot(wHb, self.bHi)

        new_frame = cv2.warpPerspective(image, self.wHi, self.size, flags=cv2.INTER_NEAREST)
        #self.mosaic[new_frame > 0] = new_frame[new_frame > 0]
        self.mosaic[new_frame > 0] += self.lamb * (new_frame[new_frame > 0] - self.mosaic[new_frame > 0])

        cv2.circle(self.traj, tuple(int(x) for x in wHb[:2, 2]), 1, (0, 0, 255), -1)
        self.pix_pose = np.array([wHb[0, 2], wHb[1, 2], ang[0]])

    
    def get_map(self, trajectory=False, centered=False):
        disp = self.mosaic.copy()
        if (trajectory): 
            disp[self.traj > 0] = self.traj[self.traj > 0]
        if (centered):
            R = cv2.getRotationMatrix2D((self.pix_pose[0], self.pix_pose[1]), -180 / math.pi * self.pix_pose[2], 1)
            R[0, 2] += self.size[0] / 2 - self.pix_pose[0]
            R[1, 2] += self.size[0] / 2 - self.pix_pose[1]
            disp = cv2.warpAffine(disp, R, self.size, flags=cv2.INTER_NEAREST)
        return disp


    def get_rover_view(self, trajectory=False):
        m = self.get_mosaic(trajectory)
        disp = cv2.warpPerspective(m, np.linalg.inv(self.wHi), self.size[:2])
        return disp


    def get_homography(self, dst=(-0.5, 0.6, 1., 1.)):
        dst_yx = self.rect2pts(dst)
        dst_w = np.vstack((dst_yx[::-1,:], np.zeros(4)))
        dst_pix = self.to_pix(dst_yx)
        src_pix = aurora.imgproc.projectp(dst_w, T=aurora.rover.img2base(), K=aurora.rover.K())
        return cv2.findHomography(src_pix.T, dst_pix.T)[0]


    def compose_homography(self, x, y, theta, uv0=np.array([250, 250])):
        cy = math.cos(theta)
        sy = math.sin(theta)
        C = np.array([[ cy, sy, self.to_pix(y) + uv0[0]],
                      [-sy, cy, self.to_pix(x) + uv0[1]],
                      [  0,  0,  1]])
        return C


    def rect2pts(self, r):
        p = np.array([[r[0], r[0],      r[0]+r[2], r[0]+r[2]],
                      [r[1], r[1]+r[3], r[1]+r[3], r[1]     ]])
        return p


    def to_pix(self, meter):
        return meter / self.resolution


####################################
#  sample code                     #
####################################
if __name__ == '__main__':
    import os
    import glob
    from colorcorrect.algorithm import grey_world

    SRCDIR = '/Users/kyoheee/FieldData/MarsYard2015/bedrock01'

    vecT = aurora.localization.read_from_file(os.path.join(SRCDIR, 'vo_raw.txt'))
    vecP = aurora.localization.concat_motion(vecT, plane_assumption=1)

    mapper = Mapper((500, 500), 0.02, lamb=1)
    cv2.namedWindow("test")

    files = glob.glob(os.path.join(SRCDIR, 'img/L*.png'))
    for i, f in enumerate(files):
        if (i < 540): continue
        wTc = vecP[i, :].reshape((4, 4))

        img = cv2.imread(f)
        img = grey_world(img)
        img[:100, :, :] = 0
        #cv2.imshow('Image', img)
        #cv2.waitKey(1)


        mapper.add_image(img, wTc)
        tmap = mapper.get_map(trajectory=True, centered=True)
        tmap = np.fliplr(np.flipud(tmap))
        #cv2.imshow("im", tmap)
        cv2.imwrite('/Users/kyoheee/Codes/aurora/gui/static/data/tmap_tmp.png', tmap)
        os.rename('/Users/kyoheee/Codes/aurora/gui/static/data/tmap_tmp.png', '/Users/kyoheee/Codes/aurora/gui/static/data/tmap.png')
        cv2.waitKey(500)


    #update_texture(img, mymap, np.eye(4))
    #mymap = Map((500, 500), 0.02, depth=3)
    #plt.figure()
    #mymap.imshow('test')
    
    plt.pause(1)
    raw_input()  # wait key input

