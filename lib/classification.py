#!/usr/bin/env python
""" Terrain classification using an image

Input:
    - RGB rectified image

Output:
    - Terrain class image (single channel, 0-255)

Written by Kyohei Otsu <kyon@ac.jaxa.jp> in 2015-04-18
"""

import numpy as np
import cv2
import sklearn

import common


@common.runonce
def init_module():
    global classifier
    classifier = TerrainClassifier()


def classify_sand_rock(img):
    pass

def classify_grass_rock(img):
    pass



class TerrainClassifier:
    def __init__(self):
        # initialize classifiers
        pass
    
    def classify(self, img):
        return self.sand_rock(img)

    def sand_rock(self, img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = np.array(np.logical_and(img > 0, img < 60), dtype=np.uint8)
        cv2.imshow("lab", 255 * label)
        cv2.waitKey(1)
        return label

        





####################################
#  sample code                     #
####################################
if __name__ == '__main__':


    raw_input()  # wait key input


