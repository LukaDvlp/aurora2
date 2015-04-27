#!/usr/bin/env python
""" Image processing utility

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-20
"""

import numpy as np
import cv2
from sklearn import linear_model

import transformations as tfm
import rover
import common


@common.runonce
def init_module():
    global matcher
    global vd
    h, w = 480, 640
    max_disparity = 200
    matcher = StereoMatcher()
    vd = VDImage((h, max_disparity))



################################################################################
# Transformation
################################################################################

def transformp(pts, T):
    assert T.shape[0] >= 3 and T.shape[1] == 4

    N = pts.shape[1]
    new_pts = np.dot(T[:3, :3], pts) + np.tile(T[:3, 3:], N)
    return new_pts


def projectp(pts, T, K):
    pts2 = transformp(pts, T)
    impt = np.dot(K, pts2)
    return np.array([impt[0, :] / impt[2, :], impt[1, :] / impt[2, :]])


def compute_tilt(imL, imR, lamb=1):
    init_module()

    disp = matcher.dense(imL, imR, scale=0.5)
    vd.from_dense(disp)
    tilt = vd.tilt()
    if tilt is not None:
        compute_tilt.tilt += (tilt - compute_tilt.tilt) * lamb
    return compute_tilt.tilt
compute_tilt.tilt = rover.tilt


################################################################################
# Stereo Vision
################################################################################

class StereoMatcher:
    def __init__(self):
        # feature detection
        self.surf = cv2.SURF(400)
        self.orb = cv2.ORB()

        # feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # dense stereo
        self.max_d = 16 * 6  # max disparity
        self.min_d = 4       # min disparity
        self.wsize = 21      # sad window size
        self.stereo = cv2.StereoSGBM(self.min_d, self.max_d, self.wsize)


    @common.timeit
    def sparse(self, imL, imR, scale=1, algorithm=0):
        '''
            return Nx2 numpy array containing keypoint location
                algorithm: ORB(0), SURF(1)
                    ... use SURF for better results, ORB for better computation
        '''
        if scale != 1:
            new_shape = tuple(int(scale * s) for s in imL.shape[1::-1])
            imL = cv2.resize(imL, new_shape)
            imR = cv2.resize(imR, new_shape)
    
        if algorithm == 0:  # ORB
            # detect keypoints and find matches
            kpL, desL = self.orb.detectAndCompute(imL, None)
            kpR, desR = self.orb.detectAndCompute(imR, None)
            matches = self.bf.match(desL, desR)
            matches = sorted(matches, key = lambda x:x.distance)

            # find good matchies
            goodL = np.empty((0, 2))
            goodR = np.empty((0, 2))
            for m in matches[:len(matches)/2]:
                goodL = np.vstack((goodL, kpL[m.queryIdx].pt))
                goodR = np.vstack((goodR, kpR[m.trainIdx].pt))

        else:               # SURF
            # detect keypoints and find matches
            kpL, desL = self.surf.detectAndCompute(imL, None)
            kpR, desR = self.surf.detectAndCompute(imR, None)
            matches = self.flann.knnMatch(desL, desR, k=2)
    
            # find good matchies
            goodL = np.empty((0, 2))
            goodR = np.empty((0, 2))
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    delta = kpL[m.queryIdx].pt[0] - kpR[m.trainIdx].pt[0]
                    if delta > self.min_d * scale and delta < self.max_d * scale:
                        goodL = np.vstack((goodL, kpL[m.queryIdx].pt))
                        goodR = np.vstack((goodR, kpR[m.trainIdx].pt))

        return goodL / scale, goodR / scale
    

    @common.timeit
    def dense(self, imL, imR, scale=1):
        '''
            return dense disparity map
        '''
        if scale != 1:
            org_shape = imL.shape[1::-1]
            new_shape = tuple(int(scale * s) for s in org_shape)
            imL = cv2.resize(imL, new_shape)
            imR = cv2.resize(imR, new_shape)
            self.stereo.minDisparity = int(np.floor(self.min_d * scale))
            self.stereo.numberOfDisparities = int(np.ceil(self.max_d * scale / 16) * 16)
            self.stereo.SADWindowSize = int(np.ceil(self.wsize * scale / 2) * 2 + 1)

        if imL.ndim == 3:
            imL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
            imR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(imL, imR) / 16.
        disparity[get_mask(disparity.shape) == 0] = 0

        if (scale != 1):
            disparity = cv2.resize(disparity, org_shape)
            disparity = disparity / scale

        return disparity


################################################################################
# V-Disparity
################################################################################

class VDImage:
    def __init__(self, shape):
        # V-disparity image
        self.img = np.zeros(shape, dtype=np.uint8)
        self.min_d = 20

        # Ground estimation
        #self.model = linear_model.LinearRegression()
        self.model = linear_model.RANSACRegressor(linear_model.LinearRegression())
        self.min_count = 1


    def from_sparse(self, kpL, kpR):
        '''
            Compute V-disparity image from sparse feature matches
        '''
        self.img.fill(0)
        delta = kpL[:, 0] - kpR[:, 0]
        for v, d in zip(kpL[:, 1], delta):
            v, d = int(v), int(d)
            if 0 <= v < self.img.shape[0] and self.min_d <= d < self.img.shape[1]:
                self.img[v, d] += 1
        return self.img

    
    def from_dense(self, disparity):
        '''
            Compute V-disparity image from dense disparity map
        '''
        self.img.fill(0)
        disparity = np.array(disparity, dtype=np.int)
        disparity[:, :disparity.shape[1]/4] = 0  # remove border region
        disparity[:, -disparity.shape[1]/4:] = 0  # remove border region
        disparity[np.logical_or(disparity < self.min_d, disparity >= self.img.shape[1])] = 0 # just in case
        for i, row in enumerate(self.img):
            self.img[i] = np.bincount(disparity[i, :], minlength=self.img.shape[1])
        self.img[:, 0] = 0
        return self.img


    def compute_ground(self):
        '''
            Compute ground coefficients using ransac
            V = A(0) * D + A(1)
        '''
        img = self.img
        rows, cols = np.where(img >= self.min_count)
        if len(rows) == 0: return None
        self.model.fit(np.reshape(cols, (cols.size, 1)), rows)
        A = (self.model.estimator_.coef_[0][0], self.model.estimator_.intercept_[0])
        return A


    def tilt(self):
        '''
            Compute tilt angle [rad]
        '''
        A = self.compute_ground()
        if A is not None: 
            tilt = np.arccos(max(0, min(1, (rover.bl * A[0]) / rover.camH)))
            if 0 < tilt < np.pi / 4: return tilt
        return None

    def imshow(self, ground=False):
        img = 255. * self.img / np.amax(self.img)
        if ground:
            A = self.compute_ground()
            x0 = (0, int(A[1]))
            x1 = (int((img.shape[0] - A[1]) / A[0]), img.shape[0])
            cv2.line(img, x0, x1, 1, 1)
        cv2.imshow("V-Disparity", img)
        cv2.waitKey(1)



#############################################################################
# misc.
################################################################################

# NEEDED????
def get_mask(shape):
    '''
        return mask for region of interest
        (should adjust for each camera setting)
    '''
    h, w = shape[:2]
    margin = 20
    mask = np.zeros((h, w), np.uint8)
    mask[h/4:h-margin, margin:-margin] = 255
    return mask


################################################################################
# Sample code
################################################################################
if __name__ == '__main__':
    import os
    import time

    init_module()

    SRCDIR = '/Users/kyoheee/FieldData/MarsYard2015/bedrock01'
    imL = cv2.imread(os.path.join(SRCDIR, 'img/L000300.png'))
    imR = cv2.imread(os.path.join(SRCDIR, 'img/R000300.png'))

    # sparse stereo
    kpL, kpR = matcher.sparse(imL, imR, scale=1)
    sparse_match = imL.copy()
    for k, l in zip(kpL, kpR):
        def tuplize(arr): return tuple([int(a) for a in arr])
        cv2.line(sparse_match, tuplize(k), tuplize(l), (0, 255, 0), 1)
    cv2.imshow("sparse match", sparse_match)
    cv2.waitKey(30)

    # dense stereo
    disp = matcher.dense(imL, imR, scale=1)
    #disp = matcher.dense(imL, imR, scale=0.5)
    cv2.imshow("disparity map", disp / np.amax(disp))
    cv2.waitKey(30)


    # V-disparity
    imvd = vd.from_sparse(kpL, kpR)
    vd.imshow(ground=True)
    cv2.waitKey(-1)

    tilt = vd.tilt()
    if tilt is not None: print tilt / np.pi * 180

    imvd = vd.from_dense(disp)
    vd.imshow(ground=True)
    tilt = vd.tilt()
    if tilt is not None: print tilt / np.pi * 180

    # tilt computation
    compute_tilt(imL, imR)

    raw_input()

