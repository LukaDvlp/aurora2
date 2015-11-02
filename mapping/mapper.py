#!/usr/bin/env python
""" Mapping module using homography

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30
"""

import math
import yaml

import numpy as np
import cv2
from skimage import color

from aurora.core import core
from aurora.core import decorator
from aurora.loc import rover
from aurora.loc import transformations as tfm

# global mappers
vizmap = []
clsmap = []
hzdmap = []
elvmap = []

def setup(yamlfile):
    '''load config from yaml'''
    data = open(yamlfile).read()
    config = yaml.load(data)

    global vizmap, clsmap, hzdmap, elvmap
    vizmap = Mapper(config['vizmap'])
    clsmap = Mapper(config['clsmap'])
    hzdmap = Mapper(config['hzdmap'])
    elvmap = Mapper(config['elvmap'])

    rover.setup(config['rover_yaml'])


class Mapper():
    def __init__(self, config):
        '''
            Init Mapper instance
                config: dictionary that contains 'resolution', 'dpm', 'innovative'
        '''
        self.shape = (config['resolution'][0], config['resolution'][1])  # width, height
        self.dpm   = config['dpm']  # dots per meter
        self.lamb  = config['innovative']  # innovation ratio
        
        self.bHi = self.get_homography()
        self.center0 = np.array([s / 2 for s in self.shape[::-1]])  # [u0, v0]
        self.center  = np.array([s / 2 for s in self.shape[::-1]])  # [u0, v0]
        self.pose = np.append(self.center, [0])  # [u, v, theta]

        # images
        self.mosaic = np.zeros((self.shape[1], self.shape[0], 3), dtype=np.uint8)
        self.traj = np.zeros(self.mosaic.shape, dtype=np.uint8)
        self.init_grid()


    def add_image(self, image, pose):
        '''
            Add new image to map
        '''
        wHb, self.pose = self.compose_homography(pose)
        map_updated = self.move_map()
        if map_updated:
            wHb, self.pose = self.compose_homography(pose)
        self.wHi = np.dot(wHb, self.bHi)

        # update map
        new_frame = cv2.warpPerspective(image, self.wHi, self.shape, flags=cv2.INTER_NEAREST)
        self.mosaic[new_frame > 0] += self.lamb * (new_frame[new_frame > 0] - self.mosaic[new_frame > 0])

        # add trajectory point
        cv2.circle(self.traj, tuple(int(x) for x in self.pose[:2]), 2, (255, 0, 0), -1)
        

    def add_topview_image(self, image, pose):
        '''
            Add new topview image to map
        '''
        wHb, self.pose = self.compose_homography(pose)
        map_updated = self.move_map()
        if map_updated:
            wHb, self.pose = self.compose_homography(pose)
        #self.wHi = np.dot(wHb, self.bHi)
        self.wHi = wHb  # no need to project
        self.wHi[0, 2] -= 200
        self.wHi[1, 2] -= 200

        # update map
        new_frame = cv2.warpPerspective(image, self.wHi, self.shape, flags=cv2.INTER_NEAREST)
        self.mosaic[new_frame > 0] += self.lamb * (new_frame[new_frame > 0] - self.mosaic[new_frame > 0])

        # add trajectory point
        cv2.circle(self.traj, tuple(int(x) for x in self.pose[:2]), 2, (255, 0, 0), -1)
        

    
    def get_map(self, trajectory=False, grid=False, centered=False):
        '''
            Get curremt map
                trajectory: show rover trajectory
                centered: flag for rover-centered map
        '''
        disp = self.mosaic.copy()
        if (trajectory): 
            disp[self.traj > 0] = self.traj[self.traj > 0]
        if (grid): 
            disp[self.grid > 0] = self.grid[self.grid > 0]
        if (centered):
            R = cv2.getRotationMatrix2D(tuple(self.pose[:2]), -180 / math.pi * self.pose[2], 1)
            R[0, 2] += self.shape[0] / 2 - self.pose[0]
            R[1, 2] += self.shape[0] / 2 - self.pose[1]
            disp = cv2.warpAffine(disp, R, self.shape, flags=cv2.INTER_NEAREST)
        return disp


    def get_cmap(self, centered=False):
        cmap = 255 * np.array(np.sum(self.mosaic, axis=2) > 0, dtype=np.uint8)
        rover_pix = np.array([self.meter2pix(rover.width), self.meter2pix(rover.length)])
        d = int(np.ceil(np.linalg.norm(rover_pix)))
        rover_space = np.zeros((d, d), dtype=np.uint8)
        cv2.circle(rover_space, (d/2, d/2), d/2, 1, -1)
        cmap = cv2.dilate(cmap, rover_space)
        if (centered):
            R = cv2.getRotationMatrix2D(tuple(self.pose[:2]), -180 / math.pi * self.pose[2], 1)
            R[0, 2] += self.shape[0] / 2 - self.pose[0]
            R[1, 2] += self.shape[0] / 2 - self.pose[1]
            cmap= cv2.warpAffine(cmap, R, self.shape, flags=cv2.INTER_NEAREST)
        return cmap


    def get_rover_view(self, trajectory=False):
        '''
            Get rover-view of the map
        '''
        m = self.get_mosaic(trajectory)
        disp = cv2.warpPerspective(m, np.linalg.inv(self.wHi), self.shape[:2])
        return disp

    
    def get_pose_pix(self, pose):
        '''
            Return pose in pixels from rover coordinates
        '''
        return np.array([self.meter2pix(pose[1]) + self.center0[0],
                        self.meter2pix(pose[0]) + self.center0[1],
                        pose[2]])
    

    def get_pose_meter(self, pose):
        '''
            Return pose in meters 
        '''
        return np.array([self.pix2meter(pose[1] - self.center0[0]), 
                         self.pix2meter(pose[0] - self.center0[1]), 
                         pose[2]])


    def get_world_pose_meter(self, pose):
        '''
            Return world-coordinate pose in meters 
            TODO:
        '''
        return np.array([self.pix2meter(pose[1] - self.center[0]), 
                         self.pix2meter(pose[0] - self.center[1]), 
                         pose[2]])

    

    def get_homography(self, dst=(-0.5, 0.6, 1., 1.)):
        '''
            Compute homography between image plane and ground plane
        '''
        dst_yx = self.rect2pts(dst)
        dst_w = np.vstack((dst_yx[::-1,:], np.zeros(4)))
        dst_pix = self.meter2pix(dst_yx)
        src_pix = tfm.projectp(dst_w, T=rover.iTb, K=rover.KL)
        return cv2.findHomography(src_pix.T, dst_pix.T)[0]


    def compose_homography(self, pose):
        cy = math.cos(pose[2])
        sy = math.sin(pose[2])
        C = np.array([[ cy, sy, self.meter2pix(pose[1]) + self.center[0]],
                      [-sy, cy, self.meter2pix(pose[0]) + self.center[1]],
                      [  0,  0,  1]])
        pos = np.array([C[0, 2], C[1, 2], pose[2]])
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
        self.init_grid()
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
        print 'DEBUG(mapper): map shifted'
        return True


    def init_grid(self):
        '''Init grid (1m mesh)'''
        self.grid = np.zeros(self.mosaic.shape, dtype=np.uint8)
        for i in range(int(np.floor(self.shape[0] / self.dpm) + 1)):
            row = int(self.meter2pix(i))
            cv2.line(self.grid, (0, row), (self.shape[1], row), (0, 255, 0), 1)
        for j in range(int(np.floor(self.shape[1] / self.dpm) + 1)):
            col = int(self.meter2pix(j))
            cv2.line(self.grid, (col, 0), (col, self.shape[0]), (0, 255, 0), 1)


    def meter2pix(self, meter):
        return meter * self.dpm


    def pix2meter(self, pix):
        return 1.0 * pix / self.dpm


## Sample code
if __name__ == '__main__':
    from aurora.loc import pose2d

    # load configure from files
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    setup(core.get_full_path('config/map_config.yaml'))
    pose = pose2d.Pose2D()

    # dummy image
    im = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(im, (300, 300), 100, (255, 120, 20), -1)
    cv2.rectangle(im, (500, 210), (580, 320), (20, 22, 180), -1)
    cv2.imshow('dummy', im)
    cv2.waitKey(1)

    # dummy motion
    m = np.array([0.1, 0, 0.02])

    for i in range(100):
        p = pose.update(m)
        vizmap.add_image(im, p)

        # Normal map
        topmap = vizmap.get_map(trajectory=True, grid=True, centered=True)
        cv2.imshow('Top view', topmap)
        cv2.waitKey(1)

        # Cost map
        cmap = vizmap.get_cmap(centered=True)
        cv2.imshow('Cost map', cmap)
        cv2.waitKey(1)

        cv2.waitKey(1000)

