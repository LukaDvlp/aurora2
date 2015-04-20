#!/usr/bin/env python
""" Mapping module using homography

Input:
    - Left image

Output:
    - Terrain texture image

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-18
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2

class Map:
    '''
        Class for rover-centered 2D map
    '''
    def __init__(self, shape, gridsz, depth=1):
        shape = tuple([2 * int(np.floor(s / 2)) + 1 for s in shape[:2]])
        if depth == 1: self.map_ = np.zeros(shape)
        else:          self.map_ = np.zeros(shape + (depth,))
        self.gridsz_ = gridsz
        self.center_ = tuple([np.ceil(s / 2) for s in self.map_.shape[1::-1]])
        self.dim_ = self.get_dim_world()


    def get_dim_world(self):
        ''' 
            Return rectangle in world coordinates
            (x0, y0, width, height)
        ''' 
        dim = tuple([-c for c in self.center_]) + self.map_.shape[1::-1]
        dim = tuple([d * self.gridsz_ for d in dim])
        return dim


    def show(self):
        '''
            show map in contourf plot
        '''
        if self.map_.ndim == 3:
            logging.error('The show() function only valid for single channel map')
            return
        xarr = np.linspace(self.dim_[0], self.dim_[0] + self.dim_[2] - self.gridsz_, self.map_.shape[1])
        yarr = np.linspace(self.dim_[1], self.dim_[1] + self.dim_[3] - self.gridsz_, self.map_.shape[0])
        h = plt.contourf(xarr, yarr, self.map_)
        plt.pause(1)
        return h


    def imshow(self, wname, bgr=False):
        if bgr == True:
            assert self.map_.ndim == 3
            cv2.imshow(wname, np.flipud(self.map_[:, :, ::-1]))
        else:
            cv2.imshow(wname, np.flipud(self.map_))
        cv2.waitKey(1)

        

def update_texture(img, tex_map, pTc):

    return tex_map


####################################
#  sample code                     #
####################################
if __name__ == '__main__':

    mymap = Map((500, 201), 0.1, depth=3)
    mymap.map_[:, :4, 0] = 200
    mymap.map_[:, 4:20, 1] = 200
    mymap.map_[:, :4, 2] = 400
    plt.figure()
    mymap.show()
    mymap.imshow('test')

    plt.pause(1)
    raw_input()  # wait key input

