#!/usr/bin/env python
"""2D pose estimator

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-30
"""

from math import pi, cos, sin

import numpy as np
from aurora.loc import transformations as tfm


class Pose2D:
    def __init__(self):
        # global pose [x, y, yaw]
        self.pose = np.zeros(3)


    def pose_from_matrix(self, T):
        '''Returns pose X=[x,y,yaw] from transformation matrix'''
        ypr = tfm.euler_from_matrix(T, 'rzyx')
        xyz = T[:3,3]
        return np.array([xyz[0], xyz[1], ypr[0]])


    def matrix_from_pose(self, X):
        '''Return transformation matrix from pose'''
        T = tfm.euler_matrix(X[2], 0, 0, 'rzyx')
        T[:2, 3] = X[:2]
        return T


    def update(self, X):
        '''Update pose in 2D space
        X = [dx,dy,dyaw]
        '''
        R = np.array([[cos(self.pose[2]), -sin(self.pose[2]), 0],
                      [sin(self.pose[2]),  cos(self.pose[2]), 0],
                      [                0,                  0, 1]])
        self.pose = self.pose + np.dot(R, X)
        return self.pose


    def update_from_matrix(self, T):
        '''Update pose in 2D space from transformation matrix'''
        return self.update(self.pose_from_matrix(T))


## Sample code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pose = Pose2D()

    # generate synthetic motion
    pTc = tfm.euler_matrix(pi/16, pi/3, 0, 'rzyx')
    pTc[0, 3] = 1
    pTc[1, 3] = 2

    # concatenate motion
    pos = np.zeros((0, 3))
    for i in range(50):
        p = pose.update_from_matrix(pTc)
        print p
        pos = np.vstack((pos, p))

    # plot
    plt.figure()
    plt.axis('equal')
    plt.plot(pos[:, 0], pos[:, 1], marker='o')

    plt.pause(1)
    raw_input()

