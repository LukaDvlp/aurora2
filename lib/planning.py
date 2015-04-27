#!/usr/bin/env python
""" Path planning module

Input: 
    - top-view traversability map 
    - goal position (X, Y, [theta])

Output:
    - waypoint positions
    - next commands

Written by Kyohei Otsu <kyon@ac.jaxa.jp> in 2015-04-18
"""

from itertools import product

import logging
import numpy as np
import cv2

import astar
import rover


class Planner:
    def __init__(self, shape):
        self.scale = 0.1
        scaled_shape = tuple([int(s * self.scale) for s in shape[:2]])
        self.graph, self.nodes = astar.make_graph(scaled_shape[0], scaled_shape[1])


    def plan_path(self, cmap, start_pos, goal_pos):
        '''
            cmap: rover-centered topview class map (up=forward)
            goal_pos, start_pos: in pixels
        '''
        cmap = cv2.resize(cmap, tuple([int(s * self.scale) for s in cmap.shape[:2]]))
        start_pos = tuple([int(p * self.scale) for p in start_pos])
        goal_pos  = tuple([int(p * self.scale) for p in goal_pos])
    
        costs = self.convert_costmap(cmap)
        for i, j in product(range(cmap.shape[0]), range(cmap.shape[1])):
            self.nodes[i][j].cost = costs[i][j]

        paths = astar.AStarGrid(self.graph)
        start, end = self.nodes[start_pos[0]][start_pos[1]], self.nodes[goal_pos[0]][goal_pos[1]]
        path, score = paths.search(start, end)
        if path is None:
            logging.error("No path found")
            return None, None
        else:
            waypoints = [[p.x, p.y] for p in path]
            return np.array(waypoints) / self.scale, score



    def convert_costmap(self, cmap):
        '''
            convert multi channel terrain map to single-channel cost map
        '''
        diag_pix = int(self.scale * rover.diag() / 0.02)
        kernel = np.ones((diag_pix, diag_pix), np.uint8)
        inv_trav = np.array(cmap[:, :, 2] > 0, dtype=np.uint8)
        inv_trav = cv2.dilate(inv_trav, kernel)
        return inv_trav



####################################
#  sample code                     #
####################################
if __name__ == '__main__':
    tmap = cv2.imread('/Users/kyoheee/Codes/aurora/gui/static/data/cmap.png')
    planner = Planner(tmap.shape)
    wps = planner.plan_path(tmap, (250, 250), (300, 100))
    print wps


    raw_input()  # wait key input


