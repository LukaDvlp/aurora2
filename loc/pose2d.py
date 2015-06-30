#!/usr/bin/env python
"""2D pose estimator

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30
"""

from math import pi, cos, sin

import numpy as np
from aurora.loc import transformations as tfm


# global pose [x, y, yaw]
pose = np.zeros(3)


def pose_from_matrix(T):
    '''Returns pose X=[x,y,yaw] from transformation matrix'''
    ypr = tfm.euler_from_matrix(T, 'rzyx')
    xyz = T[:3,3]
    return np.array([xyz[0], xyz[1], ypr[0]])


def update(X):
    '''Update pose in 2D space
    X = [dx,dy,dyaw]
    '''
    global pose
    R = np.array([[cos(pose[2]), -sin(pose[2]), 0],
                  [sin(pose[2]),  cos(pose[2]), 0],
                  [           0,             0, 1]])
    pose = pose + np.dot(R, X)
    return pose


def update_from_matrix(T):
    '''Update pose in 2D space from transformation matrix'''
    update(pose_from_matrix(T))
    return pose


## Sample code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #T = tfm.euler_matrix(pi/4, pi/3, 0, 'rzyx')
    #print pose_from_matrix(T)

    # generate synthetic motion
    pTc = tfm.euler_matrix(pi/16, pi/3, 0, 'rzyx')
    pTc[0, 3] = 1
    pTc[1, 3] = 2

    # concatenate motion
    pos = np.zeros((0, 3))
    for i in range(50):
        update_from_matrix(pTc)
        pos = np.vstack((pos, pose))

    # plot
    plt.figure()
    plt.axis('equal')
    plt.plot(pos[:, 0], pos[:, 1], marker='o')

    plt.pause(1)
    raw_input()

