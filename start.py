#!/usr/bin/env python
"""Top-level module of aurora rover controller

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-12

Usage:
    $ python start.py
"""

import threading
import Queue
import math

import numpy as np
import cv2

from aurora.core import core
from aurora.hw import camera
from aurora.hw import obc
from aurora.loc import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d
from aurora.mapping import mapper
from aurora.nongeom import rockdetect
from colorcorrect.algorithm import grey_world
from aurora.geom import dense_stereo
from aurora.planning import local_planner


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
    camera.setup(core.get_full_path('config/camera_config.yaml'))
    obc.setup() #TODO yaml
    vo.setup(rover)
    mapper.setup(core.get_full_path('config/map_config.yaml'))
    pose = pose2d.Pose2D()


def loop():
    '''Main loop for processing new images'''
    goal = None
    wp = np.empty((0, 3))
    while True:
        # get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None: break
        #cv2.imwrite('/tmp/L{:05d}.jpg'.format(frame), imL)
        #cv2.imwrite('/tmp/R{:05d}.jpg'.format(frame), imR)
        #imL = grey_world(imL)
        #imR = grey_world(imR)
        imL, imR = camera.rectify_stereo(imL, imR)
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        # compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        #if (np.linalg.norm(pTc[:3, 3]) < 0.05):
            #pTc[:3, 3] = 0
        p = pose.update_from_matrix(pTc)
        print 'INFO(nav): X={:.2f}, Y={:.2f}, THETA={:.1f}'.format(p[0], p[1], math.degrees(p[2]))
        if goal is not None:
            print '           ---->    X={:.2f}, Y={:.2f}'.format(goal[0], goal[1])


        # detect obstacles
        rockmask = rockdetect.from_sand(imL)
        rocks = np.zeros(imL.shape, dtype=np.uint8)
        rocks[:, :, 2] = rockmask
        rocks[rocks == 0] = 1
        cv2.imshow('imO', rockmask)

        # stereo
        '''
        imD = dense_stereo.disparity(imLg, imRg)
        imD = np.array(imD / 16., dtype=np.float)
        imD[imD < 30] = 0
        xyz = dense_stereo.reconstruct(imD)
        imLc = imL.copy()
        imLc[imD < 15] = 0
        imLc[imLg > 150] = 0
        cv2.imshow('dL',imLc)
        #cv2.imshow('dR',imRg)
        cv2.imshow('disp', imD/48.)
        '''

        # update map
        mapmask = np.ones(imL.shape, dtype=np.uint8)
        mapmask[:200, :, :] = 0
        mapmask[400:, :, :] = 0
        mapmask[:, :120, :] = 0
        mapmask[:, 520:, :] = 0
        #mapmask[imLg < 50] = 0  # shadow removal
        mapper.vizmap.add_image(imL * mapmask,  p)
        mapper.hzdmap.add_image(rocks * mapmask, p)

        # fetch new goal from queue
        try: 
            g = GOALS.get_nowait()
            goal = g
        except: 
            pass

        # goal reached?
        if goal is not None:
            if abs(goal[0] - p[0]) < 0.2 and abs(goal[1] - p[1]) < 0.2:
                print '!!!!!!!!!!!!!!!!!!!!'
                print '!!  Goal reached  !!'
                print '!!!!!!!!!!!!!!!!!!!!'
                cmd_list = []
                cmd_list.append('f')
                obc.send_cmd(cmd_list)
                wp = np.empty((0, 3))
                goal = None

        if frame % 10 == 0:
            if goal is not None:
                # generate waypoints in rover coordinates
                goal_rel = np.dot(np.linalg.inv(pose.matrix_from_pose(p)), pose.matrix_from_pose(goal))
                goal_rel = (goal_rel[0, 3], goal_rel[1, 3], 0)
                wp = local_planner.compute_waypoints(mapper.hzdmap, (0, 0, 0), goal_rel)
    
                # send commands
                if wp.shape[0] > 1:
                    cmd_arc = np.linalg.norm(wp[1, :2] - wp[0, :2])
                    cmd_theta = -math.degrees(np.arctan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0]))
                    print 'CMD: ', cmd_arc, cmd_theta
                    cmd_list = []
                    '''
                    if cmd_theta > -90 and cmd_theta < 90:
                        cmd_list.append('s{:.2f}'.format(cmd_theta))
                        cmd_list.append('d{:.2f}'.format(cmd_arc))
                    else:
                        cmd_list.append('s{:.2f}'.format(-(180-cmd_theta%360)))
                        cmd_list.append('d{:.2f}'.format(-cmd_arc))
                    '''
                    if abs(cmd_theta) <= 45:
                        # steering drive
                        obc.set_turn_mode(False)
                        obc.set_steer_angle(cmd_theta)
                        #cmd_list.append('s{:.2f}'.format(cmd_theta))
                        cmd_list.append('d{:.2f}'.format(cmd_arc))
                    elif abs(cmd_theta) > 165:
                        # back
                        obc.set_turn_mode(False)
                        cmd_list.append('d{:.2f}'.format(-cmd_arc))
                    else:
                        obc.set_turn_mode(True)
                        cmd_list.append('r{:.2f}'.format(cmd_theta))
                    obc.send_cmd(cmd_list)
                else:
                    cmd_list = []
                    cmd_list.append('f')
                    obc.send_cmd(cmd_list)
                    goal = None
            

        ## display ##
        if True:
            topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
            hzdmap = mapper.hzdmap.get_map(trajectory=False, grid=False, centered=True)
            cv2.circle(topmap, (topmap.shape[1]/2, topmap.shape[0]/2), 25, (140, 20, 130), 4)
            ovlmap = topmap.copy()
            ovlmap[hzdmap[:,:,2] > 200] = hzdmap[hzdmap[:,:,2]>200]

            # waypoints
            for i in range(wp.shape[0]):
                p = mapper.hzdmap.get_pose_pix(wp[i])
                cv2.circle(ovlmap, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
                if (i > 0):
                    cv2.line(ovlmap, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (200, 0, 0), 3)
                p0 = p


            h, w = imL.shape[:2]
            imS = np.zeros((h/2, w, 3), dtype=np.uint8)
            imS[:, :w/2] = cv2.resize(imL, (w/2, h/2))
            imS[:, w/2:] = cv2.resize(imR, (w/2, h/2))
            cv2.imshow('Stereo view', imS)
            cv2.imshow('Top view', cv2.flip(cv2.flip(topmap, 1), 0))
    
            cv2.imshow('Hazard View', cv2.flip(cv2.flip(ovlmap, 1), 0))
        cv2.waitKey(100)
    

def set_new_goal(new_goal, clear_all=False):
    '''Set new goal to queue

    Args:
        new_goal: A tuple that contains (X, Y) in global coordinates.
        clear_all: If set, all previous goals are cleared.
    '''
    if clear_all:
        with GOALS.mutex: GOALS.queue.clear()
    GOALS.put(new_goal)


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        p_r = mapper.hzdmap.get_pose_meter((400-x, 400-y, 0))
        rTg = pose.matrix_from_pose(p_r)
        wTr = pose.matrix_from_pose(pose.update((0, 0, 0)))
        wTg = np.dot(wTr, rTg)
        pw = (wTg[0, 3], wTg[1, 3], 0)
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        print '!!  Goal set {:.2f}, {:.2f}   !!'.format(pw[0], pw[1])
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        set_new_goal(pw, clear_all=True)



if __name__ == '__main__':
    setup()
    #set_new_goal((6, 1, 0))
    cv2.namedWindow('Hazard View')
    cv2.setMouseCallback('Hazard View', mouse_cb)
    loop()

