#!/usr/bin/env python
""" Vision-based localization, mapping, planning

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-27

Usage:
    $ python vision.py
"""

import os
import math
import time
import yaml
import Queue

import logging
#logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG, filename='log/vision.log')
logger = logging.getLogger(__name__)

import numpy as np
import cv2
import matplotlib.pyplot as plt

from aurora.core import core
from aurora.core import server_wrapper
from aurora.core import rate
from aurora.geom import dense_stereo
from aurora.geom import dem
from aurora.hw import camera
from aurora.hw import obc
from aurora.loc import libviso2 as vo
from aurora.loc import rover
from aurora.loc import pose2d
from aurora.loc import transformations as tfm
from aurora.planning import local_planner
from aurora.mapping import mapper


class VisionServer(server_wrapper.ServerBase):
    def setup(self):
        ## read config
        toplevel_yaml = core.get_full_path('config/toplevel.yaml')
        data = open(toplevel_yaml).read() 
        yamls = yaml.load(data)

        rover.setup(core.get_full_path(yamls['rover']))
        camera.setup(core.get_full_path(yamls['camera']))
        obc.setup()
        vo.setup(rover)
        mapper.setup(core.get_full_path(yamls['mapper']))
        dem.setup(core.get_full_path(yamls['mapper']))

        self.pose = pose2d.Pose2D()

        self.rate_pl = rate.Rate(0.7, name='pipeline')

        ## Messaging
        self.sendq = Queue.Queue()

        ## Navigation
        self.goals = Queue.Queue()
        self.next_goal = None
        self.wp = np.empty((0, 3))
        self.flag_de = False
        
        self.distance = 0
        self.prev_pose = np.zeros(3)

        self.datadir = 'log/image/{}'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        os.mkdir(self.datadir)
    

    def worker(self):
        stamp = time.time()

        ## get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None or imR is None: 
            logger.error("No image found")
            self.rate_pl.sleep()
            return

        ## save image
        cv2.imwrite('{}/L{:06d}.jpg'.format(self.datadir, frame), imL)
        cv2.imwrite('{}/R{:06d}.jpg'.format(self.datadir, frame), imR)

        ## rectify, grayscale
        imL, imR = camera.rectify_stereo(imL, imR)
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        ## compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        p = self.pose.update_from_matrix(pTc)
        #print 'INFO(nav): X={:.2f}, Y={:.2f}, THETA={:.1f}'.format(p[0], p[1], math.degrees(p[2]))
        #self.sendq.put('xyh {:.3f} {:.3f} {:.3f}'.format(p[0], p[1], p[2]))
        self.distance += np.linalg.norm(self.prev_pose[:2] - p[:2])
        self.prev_pose = p
        self.sendq.put('xyhd {:.3f} {:.3f} {:.3f} {:.3f}'.format(p[0], p[1], p[2], self.distance))

        # stereo
        imD = dense_stereo.disparity(imLg, imRg)
        imD = np.array(imD / 16., dtype=np.float)
        imD[imD < 0] = 0
        #print np.unique(imD.astype(np.float32))

        _imXYZ = dense_stereo.reproject(imD.astype(np.float32), rover.Q.astype(np.float32))
        _imXYZ[_imXYZ[:, :, 2] < 0] = 0
        _imXYZ[_imXYZ[:, :, 2] > 10] = 0
        X = np.resize(_imXYZ, (_imXYZ.shape[0] * _imXYZ.shape[1], 3)).T
        Y = tfm.transformp(X, rover.bTi)
        imXYZ = np.resize(Y.T, _imXYZ.shape)
        #print imXYZ
        #print np.unique(imXYZ[:, :, 2])
        #print np.amin(imXYZ[:, :, 2]), np.amax(imXYZ[:, :, 2])
        rocks = 255 * np.array(imXYZ[:, :, 2] > 0.2, dtype=np.uint8)
        rocks[imD == 0] = 0

        # rocks
        #rocks = 255 * np.array(imLg > 200, dtype=np.uint8)

        kern = np.ones((15, 15), dtype=np.uint8)
        rocks = cv2.dilate(cv2.erode(rocks, kern), kern)
        rocks3 = np.zeros((rocks.shape[0], rocks.shape[1], 3), dtype=np.uint8)
        rocks3[rocks == 0] = 1
        rocks3[:, :, 2] = rocks
        #rocks3[rocks == 0] = 1

        # dem
        imD_mask = np.zeros(imD.shape, dtype=np.uint8)
        imD_mask[10:-10, 10:-10] = 1;
        cv2.fillPoly(imD_mask, [np.array([[90, 470], [10, 70], [0, 70], [0, 470]], dtype=np.int32)], 0, 8)
        imD *= imD_mask
        imDEM = dem.lvd(imD)
        imDEM3 = np.zeros((imDEM.shape[0], imDEM.shape[1], 3))
        imDEM3[:, :, 2] = (imDEM + 3) * 255 / 6

        ## update map
        mapmask = np.zeros(imL.shape, dtype=np.uint8)
        mapmask[200:400, 120:520, :] = 1
        mapper.vizmap.add_image(imL * mapmask, p)
        mapper.hzdmap.add_image(rocks3 * mapmask, p)
        mapper.elvmap.add_topview_image(imDEM3, p)


        print "==================="
        logger.debug('frame {} stamp {}'.format(frame, stamp))
        logger.debug('pose={}'.format(p))
        logger.debug('distance={}'.format(self.distance))
        print "self.next_goal", self.next_goal
        print "self.goals", self.goals.queue
        #print "self.wp", self.wp

        ## planning
        if self.next_goal is None:
            try:
                self.next_goal = self.goals.get_nowait()
            except:
                self.next_goal = None

        if self.next_goal is not None:
            if self.get_distance(self.next_goal, p[:2]) < 0.5:
                self.sendq.put('!  *** Goal reached *** ')
                self.sendto_obc(['f',])
                self.wp = np.empty((0, 3))
                self.next_goal = None

        if self.wp.shape[0] == 0 and self.next_goal is not None:
            # generate waypoints in rover relative coordinates
            self.origin = p
            goal_rel = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(self.next_goal))
            goal_rel = (goal_rel[0, 3], goal_rel[1, 3], 0)
            self.wp = local_planner.compute_waypoints(mapper.hzdmap, (0, 0, 0), goal_rel)

        if frame % 1 == 0 and self.wp.shape[0] > 0:
            # compute pose of next waypoint in local frame
            wp_rel = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(self.wp[0, :]))
            if self.get_distance(self.wp[0, :], np.zeros(3)) < 0.5:
                self.wp = np.delete(self.wp, 0, axis=0)

            # generate command
            if self.wp.shape[0] > 0:
                next_wp_rel = self.wp[0, :].ravel()
                oTp = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(p))
                p_rel = self.pose.pose_from_matrix(oTp)
                #print next_wp_rel
                #print p_rel
                cmd_arc = self.get_distance(next_wp_rel, p_rel)
                cmd_theta = -math.degrees(np.arctan2(next_wp_rel[1] - p_rel[1], next_wp_rel[0] - p_rel[0]))
                logger.debug('R={:.2f} THETA={:.2f}'.format(cmd_arc, cmd_theta))

                cmd_list = []
                if abs(cmd_theta) > 165:
                    # back
                    obc.set_turn_mode(False)
                    cmd_list.append('d{:.2f}'.format(-cmd_arc))
                elif abs(cmd_theta) > 90:
                    # don't handle backwards waypoint. shutdown planning
                    obc.set_turn_mode(False)
                    cmd_list.append('f')
                    logger.warn('  *** AKI never goes back! (Planning canceled) ***')
                    self.next_goal = None
                elif abs(cmd_theta) > 45:
                    # back step
                    obc.set_turn_mode(False)
                    obc.set_steer_angle(0)
                    #cmd_list.append('d{:.2f}'.format(-cmd_arc))
                    cmd_list.append('d{:.2f}'.format(-1))
                else: #if abs(cmd_theta) <= 45:
                    # steering drive
                    obc.set_turn_mode(False)
                    obc.set_steer_angle(cmd_theta)
                    cmd_list.append('d{:.2f}'.format(cmd_arc))
                '''
                else:
                    # Inspot turn
                    obc.set_turn_mode(True)
                    cmd_list.append('r{:.2f}'.format(cmd_theta))
                '''
                self.sendto_obc(cmd_list)



        ## save images 
        if frame % 1 == 0:
            topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
            cv2.circle(topmap, (topmap.shape[1]/2, topmap.shape[0]/2), 25, (140, 20, 130), 4)

            elvmap = mapper.elvmap.get_map(trajectory=True, grid=False, centered=True)
            #print np.unique(elvmap)
            hzdmap = mapper.hzdmap.get_map(trajectory=True, grid=True, centered=True)

            ovlmap = topmap.copy()
            ovlimL = imL.copy()

            ovlmap[hzdmap[:,:,2] > 200] = hzdmap[hzdmap[:,:,2]>200]

            # draw waypoints
            if self.wp.shape[0] > 0:
                p0 = mapper.vizmap.get_pose_pix(np.zeros(3))
                a0 = (imL.shape[1] / 2, 1.5 * imL.shape[0])
                wTo = self.pose.matrix_from_pose(self.origin)
                wTr = self.pose.matrix_from_pose(self.pose.update((0, 0, 0)))
                for i in range(self.wp.shape[0]):
                    oTwp = self.pose.matrix_from_pose(self.wp[i])
                    rTwp = np.dot(np.dot(np.linalg.inv(wTr), wTo), oTwp)
                    wp_r = self.pose.pose_from_matrix(rTwp)

                    # topview
                    p = mapper.vizmap.get_pose_pix(wp_r)
                    cv2.line(ovlmap, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (200, 0, 0), 3)
                    cv2.circle(ovlmap, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                    p0 = p

                    # left
                    pp = np.array([[wp_r[0]], [wp_r[1]], [0]])
                    if wp_r[0] > 0.5:
                        a = tfm.projectp(pp, rover.KL, rover.iTb)
                        cv2.line(ovlimL, (int(a0[0]), int(a0[1])), (int(a[0]), int(a[1])), (200, 0, 0), 6)
                        cv2.circle(ovlimL, (int(a[0]), int(a[1])), 8, (0, 0, 255), -1)
                        a0 = a

                for g in self.goals.queue:
                    wTg = self.pose.matrix_from_pose(g)
                    rTg = np.dot(np.linalg.inv(wTr), wTg)
                    g_rel = self.pose.pose_from_matrix(rTg)
                    p = mapper.vizmap.get_pose_pix(g_rel)
                    cv2.circle(ovlmap, (int(p[0]), int(p[1])), 5, (200, 0, 255), -1)



            datadir = core.get_full_path('viz/static/img')
            cv2.imwrite(os.path.join(datadir, '_images_left.png'), ovlimL)
            cv2.imwrite(os.path.join(datadir, '_images_visual_map.png'), cv2.flip(cv2.flip(ovlmap, 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_disparity.png'), cv2.resize(255 * plt.cm.jet(imD.astype(np.uint8)), (320, 240)))
            cv2.imwrite(os.path.join(datadir, '_images_elev_map.png'), cv2.flip(cv2.flip(elvmap[:, :, 2].astype(np.uint8), 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_hazard_map.png'), cv2.flip(cv2.flip(hzdmap[:, :, 2].astype(np.uint8), 1), 0))
            #cv2.imwrite(os.path.join(datadir, '_images_elev_map.png'), cv2.flip(cv2.flip(255 * plt.cm.jet(elvmap[:, :, 2].astype(np.uint8)), 1), 0))
            #cv2.imwrite(os.path.join(datadir, '_images_elev_map.png'), 
            #cv2.flip(cv2.transpose(cv2.flip(255 * plt.cm.jet(imDEM.astype(np.uint8)), 1)), 1))


        if frame % 15 == 0:
            self.wp = np.empty((0, 3))


        self.rate_pl.sleep()


    def handler(self, msg):
        # send
        while not self.sendq.empty():
            m = self.sendq.get()
            self.sock.send(m + "\n")
            #time.time(0.01)

        # recv
        arr = msg.split('\n')[0].split(' ')
        if arr[0] == 'd':
            self.flag_de = True
            print 'drive enable'
        elif arr[0] == 'f':
            self.sendto_obc(['f0',])
            self.flag_de = False 
            print 'drive disable'
        elif arr[0] == 'g':
            uv = np.array([float(arr[1]), float(arr[2])])
            self.set_goal_from_pixel(uv)
            print 'goal set to {}'.format(arr[1:])
        elif arr[0] == 'c':
            print 'goal clear'
            self.sendto_obc(['f0',])
            self.goals = Queue.Queue()
            self.next_goal = None


    def finalize(self):
        print "Bye"
        pass


    def sendto_obc(self, cmd_list):
        if self.flag_de:
            obc.send_cmd(cmd_list)


    def get_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


    def set_goal_from_pixel(self, uv):
        p_r = mapper.hzdmap.get_pose_meter((400-uv[0], 400-uv[1], 0))
        rTg = self.pose.matrix_from_pose(p_r)
        wTr = self.pose.matrix_from_pose(self.pose.update((0, 0, 0)))
        wTg = np.dot(wTr, rTg)
        pw = (wTg[0, 3], wTg[1, 3], 0)
        self.goals.put(pw)



## Sample code
if __name__ == '__main__':

    server_wrapper.start(("localhost", 7777), VisionServer)

