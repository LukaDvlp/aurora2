#!/usr/bin/env python
"""Test rock detection

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date 2015-07-03

Usage:
    $ python test_rockdetect.py
"""

import cv2

from aurora.nongeom import rockdetect


## Sample code
if __name__ == '__main__':

    # load image
    im = cv2.imread('img/rock.jpg')

    # detect rock
    rockmask = rockdetect.from_grass(im)

    # display
    disp = im.copy();
    disp[:, :, 2] += rockmask / 5

    cv2.imshow('Input', im)
    cv2.imshow('Output', disp)
    cv2.waitKey(-1)

