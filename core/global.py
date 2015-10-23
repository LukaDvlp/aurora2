#!/usr/bin/env python
"""Global variables

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-23
"""

import numpy as np


''' 
    Goal pose
    - Queue of numpy arrays that contains (x, y, theta)
'''
GOALS = Queue.Queue()


'''
    Measurement from ADConverter
    - Queue of numpy arrays
'''
ADC_MEAS = Queue.Queue()
ADC_MEAS.put(np.zeros(16))  # initial value


'''
    Global pose computed from visiual odometry
    - Queue of numpy arrays that contains (x, y, theta)
'''
POSE = Queue.Queue()
POSE.put(np.zeros(3))  # initial value

