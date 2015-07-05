#!/usr/bin/env python
"""Resource definition for IMU

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-05
"""

from aurora.srv.resources.float_array import FloatArray


class Acc(FloatArray):
    uri  = '/sensors/acc'


class Gyro(FloatArray):
    uri  = '/sensors/gyro'
    
    
class Inc(FloatArray):
    uri  = '/sensors/inc'
