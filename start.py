#!/usr/bin/env python
"""Top-level module of aurora rover controller

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-12

Usage:
    $ python start.py
"""

import threading
import Queue

import numpy as np
import cv2

from aurora.core import core
from aurora.hw import camera
from aurora.loc import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d
from aurora.mapping import mapper


## Global variables

# Goals
GOALS = Queue.Queue()

pose = []


## Functions for image processing

def setup():
    '''Execute initialization procedure

    This method should not block.
    '''
    global pose
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    camera.setup(core.get_full_path('config/camera_local_config.yaml'))
    vo.setup(rover)
    mapper.setup(core.get_full_path('config/map_config.yaml'))
    pose = pose2d.Pose2D()

def loop():
    '''Main loop for processing new images'''
    goal = None
    while True:
        # get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None: break
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        # compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        p = pose.update_from_matrix(pTc)

        # update map
        imLmask = imL
        imLmask[:100, :, :] = 0
        imLmask[imLg < 50] = 0  # shadow removal
        mapper.vizmap.add_image(imLmask, p)

        # fetch new goal from queue
        if goal is None:
            try: 
                goal = GOALS.get_nowait()
            except: 
                pass

        # goal reached?

        if goal is not None:
            # detect obstacles & update hazard map
            # generate waypoints
            # send commands
            pass

        # display
        topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
        cv2.imshow('Top view', topmap)
        cv2.waitKey(1)


def set_new_goal(new_goal, clear_all=False):
    '''Set new goal to queue

    Args:
        new_goal: A tuple that contains (X, Y) in global coordinates.
        clear_all: If set, all previous goals are cleared.
    '''
    if clear_all:
        with GOALS.mutex: GOALS.queue.clear()
    GOALS.put(new_goal)


if __name__ == '__main__':
    setup()
    loop()

