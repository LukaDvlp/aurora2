#!/usr/bin/env python
"""Rover coordinate transformation helper 

TODO(kyon): write detailed description about transformation 

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-29
"""

from math import pi
import yaml

import numpy as np
from numpy.linalg import inv

from aurora.core import core, decorator
from aurora.loc import transformations as tfm



## 4x4 Transformation matrices (global)
cTi = []  # camera to image
iTc = []  # image to camera
bTc = []  # rover base to left camera
cTb = []  # left camera to rover base 
bTi = []  # rover base to left image
iTb = []  # left image to rover base
cTr = []  # left camera to right camera
rTc = []  # right camera to left camera
cTs = []  # left camera to lidar
sTc = []  # lidar to left camera

KL = []  # camera matrix for left camera
KR = []  # camera matrix for right camera
Q  = []  # camera reprojection matrix for left camera

baseline = 0  # stereo baseline
focal_length = 0  # focal length
width = 0  # rover width
length = 0  # rover length


@decorator.runonce
def setup(yamlfile):
    '''Load config from yaml'''
    # Rover settings
    data = open(yamlfile).read()
    rconfig = yaml.load(data)

    global cTi, iTc, bTc, cTb, bTi, iTb, cTr, rTc
    iTc = tfm.euler_matrix(pi / 2 + deg2rad(rconfig['tilt']), 0, pi / 2, 'rxyz')
    cTi = inv(iTc)
    bTc = tfm.translation_matrix([0.20, rconfig['baseline'] / 2, rconfig['camera_height']])
    cTb = inv(bTc)
    bTi = np.dot(bTc, cTi)
    iTb = inv(bTi)
    rTc = tfm.translation_matrix([0, rconfig['baseline'], 0])
    cTr = inv(rTc) 
    sTc = tfm.translation_matrix([0, rconfig['baseline'] / 2.0, 0])
    cTs = inv(sTc) 

    global width, length
    width = rconfig['width']
    length = rconfig['length']

    # Camera settings
    data = open(core.get_full_path(rconfig['camera_yaml'])).read()
    cconfig = yaml.load(data)

    global baseline, focal_length
    baseline = rconfig['baseline']
    focal_length = cconfig['cameraL']['f']

    global KL, KR, Q
    KL = np.array([[cconfig['cameraL']['f'], 0, cconfig['cameraL']['uv0'][0]],
                   [0, cconfig['cameraL']['f'], cconfig['cameraL']['uv0'][1]],
                   [0, 0, 1]])
    KR = np.array([[cconfig['cameraR']['f'], 0, cconfig['cameraR']['uv0'][0]],
                   [0, cconfig['cameraR']['f'], cconfig['cameraR']['uv0'][1]],
                   [0, 0, 1]])
    Q  = np.array([[1, 0, 0, -cconfig['cameraL']['uv0'][0]],
                   [0, 1, 0, -cconfig['cameraL']['uv0'][1]],
                   [0, 0, 0, focal_length],
                   [0, 0, 1.0/baseline, 0]])



def deg2rad(deg):
    '''convert degree to radians'''
    return deg / 180.0 * pi


## Sample code
if __name__ == '__main__':
    setup(core.get_full_path('config/rover_coords.yaml'))

    print 'cTi = '
    print cTi

    print 'iTc = '
    print iTc

    print 'bTc = '
    print bTc

    print 'cTb = '
    print cTb

    print 'bTi = '
    print bTi

    print 'KL = '
    print KL
