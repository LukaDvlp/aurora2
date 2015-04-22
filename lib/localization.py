#!/usr/bin/env python
"""Vision-based localization using stereo imagery

Input:
    - Stereo image

Output:
    - Pose (4x4 T matrix)

Written by Kyohei Otsu <kyon@ac.jaxa.jp> in 2015-04-18
"""

def localize(imgL, imgR):
    '''
        main function
    '''
    pass


import csv

import numpy as np

import aurora
import transformations as tfm

def read_from_file(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        vecT = np.empty((0, 4 * 4))
        for row in reader:
            vecT = np.append(vecT, [[float(r) for r in row[1:17]]], axis=0)
    return vecT


def concat_motion(vecT, pos=(0, -1), plane_assumption=0):
    wTc = aurora.rover.base2cam()
    vecP = wTc.flatten()
    for row in vecT[pos[0]:pos[1]]:
        T = row.reshape((4, 4))
        wTc = np.dot(np.dot(np.dot(wTc, aurora.rover.cam2img()), T), np.linalg.inv(aurora.rover.cam2img()))
        # = wTc * cam2img() * T * cam2img().inv()

        if plane_assumption:
            wTc[2, 3] = 0.6
            r = get_angle(wTc)
            newR = tfm.euler_matrix(r[0], 0, 0, 'rzyx')
            wTc[:3, :3] = newR[:3, :3]
        vecP = np.vstack((vecP, wTc.flatten()))
    return vecP


def get_angle(wTc): 
    return tfm.euler_from_matrix(wTc, 'rzyx')


def get_position(wTc): 
    return wTc[:3, 3].copy()


####################################
#  sample code                     #
####################################
if __name__ == '__main__':


    raw_input()  # wait key input

