#!/usr/bin/env python
"""Global variables

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-23
"""

import numpy as np
import Queue


''' 
    Goal pose
    - Queue of numpy arrays that contains (x, y, theta)
'''
goals = Queue.Queue()


'''
    Measurement from ADConverter
    - Queue of numpy arrays
'''
adc_meas = Queue.Queue()
adc_meas.put(np.zeros(16))  # initial value


'''
    Global pose computed from visiual odometry
    - Queue of numpy arrays that contains (x, y, theta)
'''
pose = Queue.Queue()
pose.put(np.zeros(3))  # initial value

