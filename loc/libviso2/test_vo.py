#!/usr/bin/env python
"""Test visual odometry

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30

Configure config/rover_coords.yaml and config/camera_(local_)config.yaml

Usage:
    python test_vo.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
    
import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d as pose
from aurora.hw import camera
from aurora.core import core

## Sample code
if __name__ == '__main__':
    # load configure from files
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    camera.setup(core.get_full_path('config/camera_local_config.yaml'))
    vo.setup(rover)

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

        T = vo.update_stereo(imLg, imRg)
        p = pose.update_from_matrix(T)
        plt.plot(p[0], p[1], marker='o')
        plt.pause(1)
        print p
        pos = np.vstack((pos, p))

    # plot
    plt.figure()
    plt.axis('equal')
    plt.plot(pos[:, 0], pos[:, 1], marker='o')

    plt.pause(1)
    raw_input()  # wait key input

