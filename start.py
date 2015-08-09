#!/usr/bin/env python
"""Top-level module of aurora rover controller

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-12

Usage:
    $ python start.py
"""

import math
import threading
import time
import os
import Queue
import requests
import socket
import threading
import signal
import sys

import numpy as np
import cv2

from aurora.core import core
from aurora.core import rate
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
from aurora.demo import itokawa


## Global variables

# Goals
GOALS = Queue.Queue()

pose = []

term_flag = False


## Functions for image processing

def setup():
    '''Execute initialization procedure

    This method should not block.
    '''
    global pose
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    camera.setup(core.get_full_path('config/camera_config.yaml'))
    obc.setup() #TODO make it yaml
    vo.setup(rover)
    mapper.setup(core.get_full_path('config/map_config.yaml'))
    pose = pose2d.Pose2D()


def loop():
    '''Main loop for processing new images'''
    goal = None
    wp = np.empty((0, 3))
    rate_pl = rate.Rate(0.7, name='pipeline')
    while not term_flag:
        stamp = time.time()

        # get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None or imR is None: 
            print 'ERROR(cam): Failed to get image'
            rate_pl.sleep()
            continue
        cv2.imwrite('/tmp/L{:05d}.jpg'.format(frame), imL)
        cv2.imwrite('/tmp/R{:05d}.jpg'.format(frame), imR)
        #imL = grey_world(imL)
        #imR = grey_world(imR)
        imL, imR = camera.rectify_stereo(imL, imR)
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        # compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        p = pose.update_from_matrix(pTc)
        print 'INFO(nav): X={:.2f}, Y={:.2f}, THETA={:.1f}'.format(p[0], p[1], math.degrees(p[2]))
        if goal is not None:
            print '           ---->    X={:.2f}, Y={:.2f}'.format(goal[0], goal[1])


        # detect obstacles
        rockmask = rockdetect.from_sand(imL)
        #rockmask = rockdetect.from_grass_lpf(imL)
        #rockmask2 = rockdetect.from_grass_b(imL)
        #rockmask = np.array(np.logical_or(rockmask, rockmask2) * 255, dtype=np.uint8)
        #rockmask = cv2.medianBlur(rockmask, 21)
        rocks = np.zeros(imL.shape, dtype=np.uint8)
        rocks[:, :, 2] = rockmask
        rocks[rocks == 0] = 1

        # stereo
        imD = dense_stereo.disparity(imLg, imRg)
        imD = np.array(imD / 16., dtype=np.float)


        #shadowmask = np.array(imLg < 60, dtype=np.uint8) * 1
        #shadowmask[:3*shadowmask.shape[0]/4, :] = 0

        # update map
        mapmask = np.zeros(imL.shape, dtype=np.uint8)
        mapmask[200:400, 120:520, :] = 1
        #mapmask[shadowmask > 0] = [0, 0, 0]
        mapper.vizmap.add_image(imL * mapmask, p)
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

        if frame % 5 == 0:
            if goal is not None:
                # generate waypoints in rover coordinates
                goal_rel = np.dot(np.linalg.inv(pose.matrix_from_pose(p)), pose.matrix_from_pose(goal))
                goal_rel = (goal_rel[0, 3], goal_rel[1, 3], 0)
                wp = local_planner.compute_waypoints(mapper.hzdmap, (0, 0, 0), goal_rel)
    
                # send commands
                if wp.shape[0] > 1:
                    cmd_arc = np.linalg.norm(wp[1, :2] - wp[0, :2])
                    cmd_theta = -math.degrees(np.arctan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0]))
                    print 'INFO(path): R={:.2f} THETA={:.2f}'.format(cmd_arc, cmd_theta)
                    cmd_list = []
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
                        # Inspot turn
                        obc.set_turn_mode(True)
                        cmd_list.append('r{:.2f}'.format(cmd_theta))
                    obc.send_cmd(cmd_list)
                else:
                    cmd_list = []
                    cmd_list.append('f')
                    obc.send_cmd(cmd_list)
                    goal = None
            

        # detect itokawa
        #if frame % 5 == 0:
        if False: #goal is None:
            itokawa_mat = itokawa.detect_watanabe(imL)
            print itokawa_mat
            if itokawa_mat[0] > 0:
            	cv2.circle(imL, tuple([p for p in itokawa_mat]), 20, (0, 255, 0), 3)
            	ang = (itokawa_mat[0] - 240.) / 240. * (50. / 180 * math.pi)
            	p_r = (2., 2 * math.tan(ang), 0)
                print 'rel target: ', p_r

                rTg = pose.matrix_from_pose(p_r)
                wTr = pose.matrix_from_pose(pose.update((0, 0, 0)))
                wTg = np.dot(wTr, rTg)
                pw = (wTg[0, 3], wTg[1, 3], 0)
                set_new_goal(pw, clear_all=True)
                print '###################   ITOKAWA DETECTED ##########################'
                print 'itokawa_pos', itokawa_mat
                print 'itokawa_ang', ang
            


        ### Generate images ###
        #if True:
        if frame % 5 == 0:
            # topview map
            topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
            cv2.circle(topmap, (topmap.shape[1]/2, topmap.shape[0]/2), 25, (140, 20, 130), 4)
           
            # topview obstacle map
            hzdmap = mapper.hzdmap.get_map(trajectory=False, grid=False, centered=True)
            ovlmap = topmap.copy()
            ovlmap[hzdmap[:,:,2] > 200] = hzdmap[hzdmap[:,:,2]>200]

            if goal is not None:
                # waypoints
                for i in range(wp.shape[0]):
                    p = mapper.hzdmap.get_pose_pix(wp[i])
                    cv2.circle(ovlmap, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
                    if (i > 0):
                        cv2.line(ovlmap, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (200, 0, 0), 3)
                    p0 = p

            # save images
            datadir = core.get_full_path('viz/static/img')
            cv2.imwrite(os.path.join(datadir, '_images_visual_map.png'), cv2.flip(cv2.flip(ovlmap, 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_cost_map.png'), cv2.flip(cv2.flip(ovlmap, 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_left.png'), cv2.resize(imL, (imL.shape[1]/2, imL.shape[0]/2)))
            #cv2.imwrite(os.path.join(datadir, '_images_right.png'), imR)

            def put_img(uri, im, timestamp):
                #data = cv2.imencode('.png', im)[1].tostring().encode('base64')
                data = cv2.cv.EncodeImage('.png', cv2.cv.fromarray(im)).tostring().encode('base64')
                requests.put('http://192.168.201.10:5000/{}'.format(uri), data={'data': data, 'timestamp': timestamp})
            #put_img('/images/visual_map', topmap, frame)

        rate_pl.sleep()
        #rate_pl.report()

    

def set_new_goal(new_goal, clear_all=False):
    '''Set new goal to queue

    Args:
        new_goal: A tuple that contains (X, Y) in global coordinates.
        clear_all: If set, all previous goals are cleared.
    '''
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print '!!  Goal set {:.2f}, {:.2f}   !!'.format(new_goal[0], new_goal[1])
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
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
        set_new_goal(pw, clear_all=True)



def client_thread(csock):
    while True:
        data = csock.recv(1024)
        if not data: break
        if data[:2] == 'xy':
            xy = tuple([float(s) for s in data[2:].split(',')])
            #print xy
            p_r = mapper.hzdmap.get_pose_meter((400-xy[0], 400-xy[1], 0))
            rTg = pose.matrix_from_pose(p_r)
            wTr = pose.matrix_from_pose(pose.update((0, 0, 0)))
            wTg = np.dot(wTr, rTg)
            pw = (wTg[0, 3], wTg[1, 3], 0)
            set_new_goal(pw, clear_all=True)
    csock.close()


def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 5557))
    sock.listen(1)
    sock.settimeout(5)

    print 'INFO(comm): Starting socket server'

    while True:
        if term_flag: break

        try:
            csock, caddr = sock.accept()
        except:
            continue

        print 'INFO(comm): Accept from {}'.format(caddr)
        th_cl = threading.Thread(target=client_thread, args=(csock,))
        th_cl.start()
    



def signal_handler(signal, frame):
    print 'Terminating.....'
    global term_flag
    term_flag = True
    sys.exit(0)
    sock.close()


if __name__ == '__main__':
    setup()
    th_cmd = threading.Thread(target=start_server)
    th_cmd.start()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    #set_new_goal((6, 1, 0))
    #cv2.namedWindow('Hazard View')
    #cv2.setMouseCallback('Hazard View', mouse_cb)
    loop()

