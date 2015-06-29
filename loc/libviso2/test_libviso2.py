#!/usr/bin/env python
"""Test libviso2 module

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-13

Usage:
    python test_libviso2.py
"""

import numpy as np
import cv2

import libviso2 as vo
from aurora.loc import rover
from aurora.core import core

## Sample code
if __name__ == '__main__':
    imLp = cv2.imread("libviso2/img/I1p.png", 0)
    imRp = cv2.imread("libviso2/img/I2p.png", 0)
    imLc = cv2.imread("libviso2/img/I1c.png", 0)
    imRc = cv2.imread("libviso2/img/I2c.png", 0)

    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    vo.setup(rover)

    print vo.update_stereo(imLp, imRp)
    print vo.update_stereo(imLc, imRc)
    print vo.update_stereo(imLc, imRc)
    print vo.update_stereo(imLc, imRc)
    print vo.update_stereo(imLp, imRp)

    print vo.update_mono(imLp)
    print vo.update_mono(imLc)

    raw_input()  # wait key input

