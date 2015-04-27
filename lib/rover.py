#!/usr/bin/env python
"""Transformation util for rover geometry

Description:
    Transform direction 
        base2cam()  ==  bTc
    
    
    Written by Kyohei Otsu <kyon@ac.jaxa.jp> on 2015-03-15
"""

import math

import numpy as np

import transformations as tfm


inv = np.linalg.inv
pi = math.pi

#########################################################################
# Rover Parameters

# Camera height from ground [m]
camH = 0.6

# Camera tilt [rad] (initial value)
tilt = 25.0 / 180.0 * pi

# Stereo baseline [m]
bl = 0.105

# Camera focal length [pixel]
f = 487.7632 / 0.8

# Image center point [pixel]
uv0 = [319.5, 239.5]

# Distortion matrix
distortion = [0, 0, 0]

# Wheelbase (distance between the centers of two wheels) [m]
wb = 0.50

# Wheel width [m]
ww = 0.1

# Rover diagonal length [m]
rdiag = 0.6 

#########################################################################

def base2cam(): return tfm.translation_matrix([0, 0, camH])
def cam2base(): return inv(base2cam())

def img2cam(): return tfm.euler_matrix(pi / 2 + tilt, 0, pi / 2, "rxyz")
def cam2img(): return inv(img2cam())

def base2img(): return np.dot(base2cam(), cam2img())
def img2base(): return inv(base2img())

def cam2whlL(): return tfm.translation_matrix([0, -0.5 * bl + 0.5 * wb, -camH])
def whlL2cam(): return inv(cam2whlL())

def cam2whlR(): return tfm.translation_matrix([0, -0.5 * bl - 0.5 * wb, -camH])
def whlR2cam(): return inv(cam2whlR())

def whl2egL(): return tfm.translation_matrix([0, 0.5 * ww, 0])
def whl2egR(): return tfm.translation_matrix([0, -0.5 * ww, 0])

def diag(): return rdiag;


def K():
    return np.array([[f, 0, uv0[0]],
                     [0, f, uv0[1]],
                     [0, 0, 1     ]])

def D():
    return np.array(distortion)



if __name__ == "__main__":
    pass
    #print cam2img()
    #print K()
