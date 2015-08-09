#!/usr/bin/env python
"""Test rock detection

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date 2015-07-03

Usage:
    $ python test_rockdetect.py
"""
import sys
sys.settrace

import numpy as np
import cv2
import os

from aurora.core import core, decorator
from aurora.hw import camera
from aurora.demo import itokawa


## Sample code
if __name__ == '__main__':

    # load image
    camera.setup(core.get_full_path('config/camera_config.yaml'))
    frame, im = camera.get_mono_image()

    # detect itokawa
    itokawa_mat = itokawa.detect_watanabe(im)

    # display
    disp = im.copy()
    cv2.circle(disp, tuple([p for p in itokawa_mat]), 3, (0, 255, 0))
    datadir = core.get_full_path('viz/static/img')
    cv2.imwrite(os.path.join(datadir, '_images_left.png'), cv2.resize(disp, (disp.shape[1]/2, disp.shape[0]/2)))
    cv2.imwrite(os.path.join(datadir, '_images_cost_map.png'), disp)
    #cv2.imshow('itokawa', disp)

    cv2.waitKey(-1)

