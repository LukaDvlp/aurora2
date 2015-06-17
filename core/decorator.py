#!/usr/bin/env python
"""Useful decorators

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-12

Usage:
    @decorator.timeit
    def func_to_time():
        ...
"""

import time
import functools
import numpy as np

def timeit(func):
    '''Compute execution time for a function
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
    '''Run function only once
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if not newfunc.has_called:
            newfunc.has_called = True
            return func(*args, **kwargs)
        return None
    newfunc.has_called = False
    return newfunc


## Sample code
if __name__ == '__main__':
    @timeit
    def some_func():
        time.sleep(3)

    some_func()

