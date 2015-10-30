#!/usr/bin/env python
"""Local path planning (obstacle avoidance) module

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-02
"""

from itertools import product
import logging

import numpy as np
import cv2

from aurora.planning import astar
from aurora.core import decorator
from aurora.mapping import mapper


def compute_waypoints(hzdmap, start, goal):
    '''Compute waypoints given cost map and start/goal coordinates

    Args:
        hzdmap: Mapper module that contains hazard information
        start : current pose in world frame
        goal  : target pose in world frame
    Returns:
        wp    : List of waypoints in world frame
    '''
    cmap = hzdmap.get_cmap(centered=True)
    start_pix = hzdmap.get_pose_pix(start)
    goal_pix = hzdmap.get_pose_pix(goal)

    #cv2.imshow('cmap', cv2.flip(cv2.flip(cmap, 1), 0))

    # scaling for efficiency. adjust scale value if needed
    scale = 0.1
    cmap = cv2.resize(cmap, (int(cmap.shape[1] * scale), int(cmap.shape[0] * scale)))
    start_pix = np.array([int(scale * p) for p in start_pix])
    goal_pix = np.array([int(scale * p) for p in goal_pix])

    # perform astar pathplanning
    wp_pix = _compute_waypoints(cmap, start_pix, goal_pix)
    if (wp_pix is None):
        return np.empty((0, 3))
    wp_pix = np.array([1.0 * p / scale for p in wp_pix])

    # compute waypoints in real world
    wp = np.empty((0, 3))
    wp = np.vstack((wp, start))
    for i in range(wp_pix.shape[0]):
        p = hzdmap.get_pose_meter((wp_pix[i][0], wp_pix[i][1], 0))
        if wp.shape[0] >= 2:
            # skip on-line elements
            if abs((wp[-2, 1] - wp[-1, 1]) * (p[0] - wp[-1, 0]) - (wp[-2, 0] - wp[-1, 0]) * (p[1] - wp[-1, 1])) < 0.001:
                wp[-1, :] = p
            else:
                wp = np.vstack((wp, p))
        else:
            # skip close waypoints
            if (np.linalg.norm(p) > 0.2):
                wp = np.vstack((wp, p))

    '''
    cmap = cv2.resize(cmap, (int(cmap.shape[1] / scale), int(cmap.shape[0] / scale)))
    for i in range(wp.shape[0]):
        p = mapper.hzdmap.get_pose_pix(wp[i])
        cv2.circle(cmap, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
        if (i > 0):
            cv2.line(cmap, (int(p0[0]), int(p0[1])), (int(p[0]), int(p[1])), (200, 0, 0), 3)
        p0 = p

    cv2.imshow('cmap', cv2.flip(cv2.flip(cmap, 1), 0))
    cv2.waitKey(1)
    '''
    return wp


def _compute_waypoints(cmap, spix, gpix):
    graph, nodes = astar.make_graph(cmap.shape[0], cmap.shape[1])
    for i, j in product(range(cmap.shape[0]), range(cmap.shape[1])):
        nodes[i][j].cost = cmap[i][j]

    paths = astar.AStarGrid(graph)
    try:
        snode = nodes[int((spix[1]))][int((spix[0]))]
        gnode = nodes[int((gpix[1]))][int((gpix[0]))]
        print 'DEBUG(planner): snode', spix, snode.cost
        print 'DEBUG(planner): gnode', gpix, gnode.cost
        path, score = paths.search(snode, gnode)
        if path is not None:
            waypoints = [[p.y, p.x] for p in path]
            return np.array(waypoints)
        else:
            logging.error("No path found")
    except:
        logging.error("Start/Goal nodes are not valid")
    return None
    


## Sample code
if __name__ == '__main__':
    from aurora.core import core
    from aurora.loc import rover

    # load configure from files
    rover.setup(core.get_full_path('config/rover_coords.yaml'))
    mapper.setup(core.get_full_path('config/map_config.yaml'))

    # dummy cost map
    cv2.circle(mapper.hzdmap.mosaic, (250, 300), 5, (0, 0, 255), -1)
    cv2.circle(mapper.hzdmap.mosaic, (300, 300), 5, (0, 0, 255), -1)

    # path plannin test
    start = (0, 1, 0)
    goal  = (3, 0, 0)
    print 'Path planning from', start, 'to', goal
    wp = compute_waypoints(mapper.hzdmap, start, goal)
    print 'Waypoints:\n', wp

    # display
    cmap = mapper.hzdmap.get_map(centered=True, grid=True)
    for i in range(wp.shape[0]):
        p = mapper.hzdmap.get_pose_pix(wp[i])
        cv2.circle(cmap, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
    #cv2.imshow('result', cmap)
    #cv2.waitKey(30)

    raw_input()

