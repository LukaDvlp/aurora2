#!/usr/bin/env python
"""Basic functionality

@author kyohei otsu <kyon@ac.jaxa.jp>
@date   2015-06-12
"""

import os
import aurora


def get_pkg_path():
    '''Return absolute path to Aurora package'''
    return aurora.__path__[0]


def get_full_path(filename):
    '''Return absolute path to the file relative to Aurora package'''
    return os.path.join(get_pkg_path(), filename)


## Sample code
if __name__ == '__main__':
    
    print 'Package path: ' + get_pkg_path()
    print 'Rile path:    ' + get_full_path('core/core.py')

