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

import numpy as np
import cv2
import matplotlib.pyplot as plt

from aurora.core import core
from aurora.core import server_wrapper
from aurora.core import rate
from aurora.geom import dense_stereo
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

        self.pose = pose2d.Pose2D()

        self.rate_pl = rate.Rate(0.7, name='pipeline')

        ## Messaging
        self.sendq = Queue.Queue()

        ## Navigation
        self.goals = Queue.Queue()
        self.next_goal = None
        self.wp = np.empty((0, 3))
        self.flag_de = False
    

    def worker(self):
        stamp = time.time()

        ## get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None or imR is None: 
            print "No image found"
            self.rate_pl.sleep()
            return
        #imL, imR = camera.rectify_stereo(imL, imR)
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        ## save image

        ## compute odometry
        pTc = vo.update_stereo(imLg, imRg)
        p = self.pose.update_from_matrix(pTc)
        #print 'INFO(nav): X={:.2f}, Y={:.2f}, THETA={:.1f}'.format(p[0], p[1], math.degrees(p[2]))
        self.sendq.put('xyh {:.3f} {:.3f} {:.3f}'.format(p[0], p[1], p[2]))

        # stereo
        imD = dense_stereo.disparity(imLg, imRg)
        imD = np.array(imD / 16., dtype=np.float)


        ## update map
        mapmask = np.zeros(imL.shape, dtype=np.uint8)
        mapmask[200:400, 120:520, :] = 1
        mapper.vizmap.add_image(imL * mapmask, p)


        print "==================="
        print "frame", frame
        print "self.p", p
        print "self.next_goal", self.next_goal
        print "self.goals", self.goals.queue
        print "self.wp", self.wp
        ## planning
        if self.next_goal is None:
            try:
                self.next_goal = self.goals.get_nowait()
            except:
                self.next_goal = None

        if self.next_goal is not None:
            if self.distance(self.next_goal, p[:2]) < 0.2:
                self.sendq.put('! Goal reached')
                self.sendto_obc(['f',])
                self.wp = np.empty((0, 3))
                self.next_goal = None

        if self.wp.shape[0] == 0 and self.next_goal is not None:
            # generate waypoints in rover relative coordinates
            self.origin = p
            goal_rel = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(self.next_goal))
            goal_rel = (goal_rel[0, 3], goal_rel[1, 3], 0)
            self.wp = local_planner.compute_waypoints(mapper.hzdmap, (0, 0, 0), goal_rel)

        if frame % 5 == 0 and self.wp.shape[0] > 0:
            # compute pose of next waypoint in relative frame
            wp_rel = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(self.wp[0, :]))
            if self.distance(self.wp[0, :], np.zeros(3)) < 0.2:
                self.wp = np.delete(self.wp, 0, axis=0)

            # generate command
            if self.wp.shape[0] > 0:
                next_wp_rel = self.wp[0, :].ravel()
                oTp = np.dot(np.linalg.inv(self.pose.matrix_from_pose(self.origin)), self.pose.matrix_from_pose(p))
                p_rel = self.pose.pose_from_matrix(oTp)
                print next_wp_rel
                print p_rel
                cmd_arc = self.distance(next_wp_rel, p_rel)
                cmd_theta = -math.degrees(np.arctan2(next_wp_rel[1] - p_rel[1], next_wp_rel[0] - p_rel[0]))
                print 'INFO(path): R={:.2f} THETA={:.2f}'.format(cmd_arc, cmd_theta)

                cmd_list = []
                if abs(cmd_theta) <= 45:
                    # steering drive
                    obc.set_turn_mode(False)
                    obc.set_steer_angle(cmd_theta)
                    cmd_list.append('d{:.2f}'.format(cmd_arc))
                elif abs(cmd_theta) > 165:
                    # back
                    obc.set_turn_mode(False)
                    cmd_list.append('d{:.2f}'.format(-cmd_arc))
                else:
                    # Inspot turn
                    obc.set_turn_mode(True)
                    cmd_list.append('r{:.2f}'.format(cmd_theta))
                self.sendto_obc(cmd_list)
            else: 
                self.wp = None



        ## save images 
        if frame % 1 == 0:
            topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
            cv2.circle(topmap, (topmap.shape[1]/2, topmap.shape[0]/2), 25, (140, 20, 130), 4)

            ovlmap = topmap.copy()
            ovlimL = imL.copy()
            # draw waypoints
            if self.wp.shape[0] > 0:
                p0 = mapper.vizmap.get_pose_pix(np.zeros(3))
                a0 = (imL.shape[1] / 2, 1.5 * imL.shape[0])
                wTr = self.pose.matrix_from_pose(self.pose.update((0, 0, 0)))
                for i in range(self.wp.shape[0]):
                    # topview
                    p = mapper.vizmap.get_pose_pix(self.wp[i])
                    cv2.circle(ovlmap, (int(p[0]), int(p[1])), 3, (255, 0, 200), -1)
                    cv2.line(ovlmap, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (200, 0, 0), 3)
                    p0 = p

                    # left
                    pp = np.array([[self.wp[0, 0]], [self.wp[0, 1]], [0]])
                    a = tfm.projectp(pp, rover.KL, np.dot(rover.iTb, wTr))
                    cv2.circle(ovlimL, (int(a[0]), int(a[1])), 8, (255, 0, 200), -1)
                    cv2.line(ovlimL, (int(a0[0]), int(a0[1])), (int(a[0]), int(a[1])), (200, 0, 0), 6)
                    print "a", a
                    a0 = a


            datadir = core.get_full_path('viz/static/img')
            cv2.imwrite(os.path.join(datadir, '_images_left.png'), ovlimL)
            cv2.imwrite(os.path.join(datadir, '_images_visual_map.png'), cv2.flip(cv2.flip(ovlmap, 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_disparity.png'), cv2.resize(255 * plt.cm.jet(imD.astype(np.uint8)), (320, 240)))


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
            self.sendto_obc(['f',])
            self.flag_de = False 
            print 'drive disable'
        elif arr[0] == 'g':
            uv = np.array([float(arr[1]), float(arr[2])])
            self.set_goal_from_pixel(uv)
            print 'goal set to {}'.format(arr[1:])
        elif arr[0] == 'c':
            print 'goal clear'


    def finalize(self):
        print "Bye"
        pass


    def sendto_obc(self, cmd_list):
        if self.flag_de:
            obc.send_cmd(cmd_list)


    def distance(self, a, b):
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

