#!/usr/bin/env python
"""Test dense_stereo module

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-18

Usage:
    python test_dense_stereo.py
"""

import numpy as np
import cv2

import dense_stereo


## Sample code
if __name__ == '__main__':
    imL = cv2.imread("img/tsukuba_l.png", 0)
    imR = cv2.imread("img/tsukuba_r.png", 0)

    # show left image
    cv2.imshow("disparity", imL)
    cv2.waitKey(1)

    # compute disparity
    imD = dense_stereo.disparity(imL, imR)
    imD_ = np.array(imD, dtype=np.uint8)

    # show disparity image
    cv2.imshow("disparity", imD_)
    cv2.waitKey(-1)


