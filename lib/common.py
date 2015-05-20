#!/usr/bin/env python
""" Common utilities

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-22
"""

import time
import functools
import numpy as np

def timeit(func):
    '''
        Compute execution time for a function
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        ret = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return ret
    return newfunc


def runonce(func):
    '''
        Run function only once
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if not newfunc.has_called:
            newfunc.has_called = True
            return func(*args, **kwargs)
        return None
    newfunc.has_called = False
    return newfunc


def jet(n):
    cmap = np.zeros((n, 3), dtype=np.uint8)
    m = n / 4
    u = 255 * np.concatenate((np.linspace(1/m, 1, num=m), np.ones(m-2), np.linspace(1, 1./m, num=m)), axis=0)
    nu = u.size
    nu3q = np.floor(3 * nu / 4)
    cmap[-nu3q:, 0] = u[:nu3q]
    cmap[n/4:n/4+nu, 1] = u
    cmap[:nu3q, 2] = u[-nu3q:]
    return cmap[:, ::-1]  # bgr


####################################
#  sample code                     #
####################################
if __name__ == '__main__':

    # Measure execution time
    @timeit
    def function_to_time():
        time.sleep(3)

    function_to_time()


    # Execute initialization code
    @runonce
    def init():
        print "initialized once"
    init()  # executed
    init()  # not executed
    init()  # not executed

    raw_input()  # wait key input

