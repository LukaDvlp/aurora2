#!/usr/bin/env python
""" Common utilities

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-22
"""

import time
import functools

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

