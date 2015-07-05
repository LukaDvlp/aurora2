#!/usr/bin/env python
"""Resource definition for rover states

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-06
"""

from aurora.srv.resources.float_array import FloatArray


class PoseVO(FloatArray):
    uri  = '/states/pose_vo'

