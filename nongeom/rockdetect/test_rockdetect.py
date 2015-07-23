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

from aurora.nongeom import rockdetect


## Sample code
if __name__ == '__main__':

    # load image
    im = cv2.imread('img/rock.jpg')

    # detect rock
    rockmask = rockdetect.from_grass(im)
    kernel = np.ones((15, 7), np.uint8)
    rockmask = cv2.erode(cv2.dilate(rockmask, kernel), kernel)
    rockmask = cv2.dilate(cv2.erode(rockmask, kernel), kernel)

    # display
    disp = im.copy()
    disp[rockmask > 0] = 0

    cv2.imshow('Input', im)
    cv2.imshow('Output', disp)
    cv2.waitKey(-1)

