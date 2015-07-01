#!/usr/bin/env python
"""Test visual odometry

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30

Configure config/rover_coords.yaml and config/camera_(local_)config.yaml

Usage:
    python test_vo.py
"""

import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pch
    
import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d_ekf as pose2d
from aurora.hw import camera
from aurora.core import core

## Sample code
if __name__ == '__main__':
    # load configure from files
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    camera.setup(core.get_full_path('config/camera_local_config.yaml'))
    vo.setup(rover)
    pose = pose2d.Pose2DEKF()
    pose.setup(core.get_full_path('config/ekf_config.yaml'))

    # perform odometry
    plt.figure()
    plt.hold('on')
    plt.axis('equal')
    pos = np.zeros((0, 3))
    for i in range(1000):
        frame, imL, imR = camera.get_stereo_images()
        if imL is None: break
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', imL)
        cv2.waitKey(1)

        # update
        T = vo.update_stereo(imLg, imRg)
        p = pose.update_from_matrix(T)
        c = pose.cov
        print 'update', c

        for j in range(1):
            # predict
            p = pose.predict()
            c = pose.cov
            #u, v = np.linalg.eig(pose.ekf.P[:2, :2])
            #u = 0.25 * np.sqrt(u)
            print 'predict only', c
            pos = np.vstack((pos, p))
            plt.plot(p[0], p[1], marker='o')
            #plt.gca().add_patch(pch.Ellipse((p[0], p[1]), u[0], u[1], math.degrees(math.atan2(v[1, 0], v[0, 0]))))
            plt.pause(1)

    # plot
    plt.figure()
    plt.axis('equal')
    plt.plot(pos[:, 0], pos[:, 1], marker='o')

    plt.pause(1)
    raw_input()  # wait key input

