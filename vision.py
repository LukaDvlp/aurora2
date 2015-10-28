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
from aurora.mapping import mapper


class VisionServer(server_wrapper.ServerBase):
    def setup(self):
        toplevel_yaml = core.get_full_path('config/toplevel.yaml')
        data = open(toplevel_yaml).read() 
        yamls = yaml.load(data)

        rover.setup(core.get_full_path(yamls['rover']))
        camera.setup(core.get_full_path(yamls['camera']))
        #obc.setup()
        vo.setup(rover)
        mapper.setup(core.get_full_path(yamls['mapper']))

        self.pose = pose2d.Pose2D()

        self.rate_pl = rate.Rate(0.7, name='pipeline')


        self.msg = ""
    

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
        print 'INFO(nav): X={:.2f}, Y={:.2f}, THETA={:.1f}'.format(p[0], p[1], math.degrees(p[2]))
        self.msg = '{:.3f} {:.3f} {:.3f}'.format(p[0], p[1], p[2])

        # stereo
        imD = dense_stereo.disparity(imLg, imRg)
        imD = np.array(imD / 16., dtype=np.float)


        ## update map
        mapmask = np.zeros(imL.shape, dtype=np.uint8)
        mapmask[200:400, 120:520, :] = 1
        mapper.vizmap.add_image(imL * mapmask, p)


        ## save images 
        if frame % 5 == 0:
            topmap = mapper.vizmap.get_map(trajectory=True, grid=True, centered=True)
            cv2.circle(topmap, (topmap.shape[1]/2, topmap.shape[0]/2), 25, (140, 20, 130), 4)

            datadir = core.get_full_path('viz/static/img')
            cv2.imwrite(os.path.join(datadir, '_images_visual_map.png'), cv2.flip(cv2.flip(topmap, 1), 0))
            cv2.imwrite(os.path.join(datadir, '_images_disparity.png'), cv2.resize(255 * plt.cm.jet(imD.astype(np.uint8)), (320, 240)))


        self.rate_pl.sleep()


    def handler(self, msg):
        self.sock.send(self.msg)
        pass


    def finalize(self):
        print "Bye"
        pass


## Sample code
if __name__ == '__main__':

    server_wrapper.start(("localhost", 7777), VisionServer)

