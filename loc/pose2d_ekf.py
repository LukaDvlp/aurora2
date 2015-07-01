#!/usr/bin/env python
"""2D pose estimator using EKF

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-29
"""

from math import cos, sin, atan2
import yaml

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

from aurora.core import core
from aurora.loc import pose2d



class Pose2DEKF(pose2d.Pose2D):
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.set_pose(np.zeros(6))
        self.dt = 1


    def setup(self, yamlfile):
        '''load config from yaml'''
        data = open(yamlfile).read()
        config = yaml.load(data)


        dt = config['dt']  # timestep

        # System function
        self.ekf.F = np.eye(6) + dt * np.array([[0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0]])

        self.ekf.Q = np.diag(config['Q'])  # System noise
        self.ekf.R = np.diag(config['R'])  # Measurement noise


    def hx(self, x):
        ''' Measurement function '''
        h = np.array([[0, 0, 0,  cos(x[2]), sin(x[2]), 0],
                      [0, 0, 0, -sin(x[2]), cos(x[2]), 0],
                      [0, 0, 0,          0,         0, 1]])
        return np.dot(h, x)


    def HJ(self, x):
        ''' Jacobian
        Note: We don't use partial derivative of h, since it has problems with trigonometric functions '''
        H = np.array([[0, 0, 0,  cos(x[2]), sin(x[2]), 0],
                    [0, 0, 0, -sin(x[2]), cos(x[2]), 0],
                    [0, 0, 0,          0,         0, 1]])
        #H = np.array([[0, 0, -sin(x[2])*x[3] + cos(x[2])*x[4],  cos(x[2]), sin(x[2]), 0],
        #              [0, 0, -cos(x[2])*x[3] - sin(x[2])*x[4], -sin(x[2]), cos(x[2]), 0],
        #              [0, 0, 0, 0, 0, 1]])
        return H


    def predict(self):
        '''Predict current pose. 
        This function should be called in dt interval.
        '''
        self.ekf.predict()
        return self.pos


    def update(self, z):
        '''Measurement update'''
        self.ekf.update(z, HJacobian=self.HJ, Hx=self.hx)
        return self.pos



    def set_pose(self, pose):
        self.ekf.x = pose

    @property
    def pos(self):
        return self.ekf.x[:3]

    @property
    def cov(self):
        return np.sqrt(np.diag(self.ekf.P)[:3])




## Sample code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pose = Pose2DEKF()

    pose.setup(core.get_full_path('config/ekf_config.yaml'))

    pose.set_pose(np.array([0, 0, 0, 2, 0, 0]))

    pos = np.zeros((0, 6))
    cnv = np.zeros((0, 6))

    z = np.array([2, 0, 0])
    for i in range(50):
        pose.update(z)
        pose.predict()
        pos = np.vstack((pos, pose.pos))
        cnv = np.vstack((cnv, pose.cov))

    z = np.array([2, 0, 0.1])
    for i in range(50):
        pose.update(z)
        pose.predict()
        pos = np.vstack((pos, pose.pos))
        cnv = np.vstack((cnv, pose.cov))


    plt.figure()
    plt.axis('equal')
    plt.plot(pos[:,0], pos[:,1], marker='o')
    plt.draw()

    plt.figure()
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(pos[:,i])

    plt.figure()
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(cnv[:,i])

    plt.pause(1)
    raw_input()


