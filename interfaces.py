#!/usr/bin/env python
""" Interfaces for the Aurora library

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-27
"""

from itertools import izip, count

import numpy as np
import cv2
import skimage
import skimage.color
from colorcorrect.algorithm import grey_world

from aurora import *


@common.runonce
def init_module():
    ########## Terrain Classification  ##########
    global classifier_
    classifier_ = classification.TerrainClassifier()

    ########## Terrain Mapping  ##########
    global viz_mapper_
    global cls_mapper_
    map_shape = (500, 500) # pix
    map_resolution = 0.02  # meter/pix
    viz_mapper_ = mapping.Mapper(map_shape, map_resolution, lamb=1)
    cls_mapper_ = mapping.Mapper(map_shape, map_resolution, lamb=0.6)

    ########## Path Planning ##########
    global planner_
    global goal_
    planner_ = planning.Planner(map_shape)
    goal_ = (8, 8)  # should have no route


def rectify_tilt(imageL, imageR):
    rover.tilt = imgproc.compute_tilt(imageL, imageR, lamb=0.6)


def load_image(path):
    '''
        General interface of loading image
            path: Path to image
            Return: RGB rectified image
    '''
    image = cv2.imread(path)
    image = grey_world(image)
    return image


def classify(image):
    '''
        Performs terrain classification
            image: RGB rectified image
            Return: Terrain class image (8bit, 3-channel, 0-255)
    '''
    init_module()
    label = classifier_.classify(image)
    label = skimage.color.label2rgb(label, bg_label=0) * 255
    return label


def register(image, label, wTc):
    '''
        Registers image and label to topview map
            img: RGB rectified image (left)
            label: Corresponding label image (single-channel)
            wTc: Transformation from camera to world (4x4 matrix)
            Return: (Terrain visual map, Terrain class map)
    '''
    init_module()

    # visual image
    image[:image.shape[0]/8, :, :] = 0
    viz_mapper_.add_image(image, wTc)
    tmap = viz_mapper_.get_map(trajectory=True, centered=True)
    tmap = np.fliplr(np.flipud(tmap))

    # label image
    label[:label.shape[0]/8, :, :] = 0
    cls_mapper_.add_image(label, wTc)
    cmap3 = cls_mapper_.get_map(centered=True)
    cmap3 = np.fliplr(np.flipud(cmap3))
    cmap = np.zeros((tmap.shape[0], tmap.shape[1], 4), dtype=np.uint8)
    cmap[:, :, :3] = cmap3
    cmap[:, :, 3] = 255 * (np.sum(cmap[:, :, :3] > 0, axis=2))  # alpha channel
    return tmap, cmap


def set_goal(goal):
    init_module()
    goal_ = goal


def plan_path(cmap, wTs, wTg):
    init_module()
    start = viz_mapper_.get_pose_pix(wTs)
    goal  = viz_mapper_.get_pose_pix(wTg)
    waypoints, score = planner_.plan_path(cmap, (start[1], start[0]), (goal[1], goal[0]));


################################################################################
# Sample code
################################################################################
if __name__ == '__main__':
    import os
    import glob
    import aurora

    # path to dataset directory
    SRCDIR = '/Users/kyoheee/FieldData/MarsYard2015/freerun03'
    DATADIR = '/Users/kyoheee/Codes/aurora/gui/static/data'

    # (tmp) load visual odometry
    vecT = localization.read_from_file(os.path.join(SRCDIR, 'vo_raw.txt'))
    vecP = localization.concat_motion(vecT, plane_assumption=1)

    files  = glob.glob(os.path.join(SRCDIR, 'img/L*.png'))
    filesR = glob.glob(os.path.join(SRCDIR, 'img/R*.png'))

    for i, f, fR in izip(count(), files, filesR):
        if i < 1540: continue
        wTc = vecP[i, :].reshape((4, 4))

        # load image
        image, imageR = aurora.load_image(f), aurora.load_image(fR)

        # Rectify tilt angle based on V-disparity method
        aurora.rectify_tilt(image, imageR)

        # vision-based classification
        label = aurora.classify(image)

        # mapping
        tmap, cmap = aurora.register(image, label, wTc)

        # save
        cv2.imwrite(os.path.join(DATADIR, 'tmap_tmp.png'), tmap)
        os.rename(os.path.join(DATADIR, 'tmap_tmp.png'), os.path.join(DATADIR, 'tmap.png'))
        cv2.imwrite(os.path.join(DATADIR, 'cmap_tmp.png'), cmap)
        os.rename(os.path.join(DATADIR, 'cmap_tmp.png'), os.path.join(DATADIR, 'cmap.png'))

        # path planning
        wTg = wTc.copy()
        #wTg[:2, 3] = goal_
        wTg[:2, 3] = 1, 0
        #aurora.plan_path(cmap, wTc, wTg)

        cv2.waitKey(500)

    raw_input()  # wait key input

