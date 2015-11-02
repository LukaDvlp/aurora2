#!/usr/bin/env python
"""Converts disparity image to a DEM (digital elevation map)

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-14

Usage:
    $ python dem.py <args>
"""

import sys
import yaml

import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt

from aurora.core import core
from aurora.core import decorator
from aurora.geom import dem_impl
from aurora.loc import rover

dem = sys.modules[__name__]

# Dem options
ranges = []
resolution = []
dpm = 0.0
grid_size = 0.0
max_disparity = 0

@decorator.runonce
def setup(yamlfile):
    '''load config from yaml'''
    data = open(yamlfile).read()
    config = yaml.load(data)

    global ranges, resolution, dpm, grid_size, max_disparity
    ranges = np.array(config['elvmap']['ranges'], dtype=np.float)
    resolution = np.array(config['elvmap']['resolution'], dtype=np.float)
    dpm = config['elvmap']['dpm']
    grid_size = config['elvmap']['lvd']['grid_size']
    max_disparity = config['elvmap']['lvd']['max_disparity']

    rover.setup(core.get_full_path(config['rover_yaml']))

    dem_impl.setup(dem, rover)

def lvd(imD):
    imDEM = dem_impl.lvd(imD)
    imDEM[imDEM == 0] = -1000
    #return cv2.flip(cv2.flip(imDEM.T, 0), 1)  # v: forward   u: left
    return imDEM.T



## Sample code
if __name__ == '__main__':
    import os
    from aurora.hw import camera
    from aurora.geom import dense_stereo

    camera.setup(core.get_full_path('config/camera_local_config.yaml'))
    setup(core.get_full_path('config/map_config.yaml'))


    # load image
    frame = 0
    frame, imL, imR = camera.get_stereo_images()
    #imL, imR = camera.rectify_stereo(imL, imR)
    cv2.imshow("Left", imL)
    imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)


    imD = dense_stereo.disparity(imLg, imRg)
    imD = cv2.medianBlur(imD, 5)
    imD = np.array(imD / 16., dtype=np.float)

    imD_mask = np.zeros(imD.shape, dtype=np.uint8)
    imD_mask[10:-10, 10:-10] = 1;
    cv2.fillPoly(imD_mask, [np.array([[90, 470], [10, 70], [0, 70], [0, 470]], dtype=np.int32)], 0, 8)
    imD *= imD_mask

    cv2.imshow("dispariy", imD / 64)
    cv2.waitKey(1)

    res = dem_impl.lvd(imD)

    #res = color.label2rgb(res)
    #cv2.imshow("Result", res)
    #cv2.waitKey(1)

    plt.figure()
    plt.imshow(res)
    plt.colorbar()

    res = res[::10, ::10]
    res[res == 0] = 100

    grad = np.zeros((40, 40))
    grad[:, :-1] = np.abs(res[:, 1:] - res[:, :-1])
    grad[:-1, :] = np.abs(res[1:, :] - res[:-1, :])
    grad = cv2.resize(grad, (400, 400), interpolation=cv2.INTER_NEAREST)
    grad[grad > 50] = np.nan
    
    plt.figure()
    plt.imshow(grad)
    plt.colorbar()
    #plt.figure()
    #plt.imshow(grad_y)


    plt.draw()
    plt.pause(1)


    raw_input()

