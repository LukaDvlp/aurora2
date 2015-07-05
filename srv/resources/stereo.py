#!/usr/bin/env python
"""Resource definition for stereo camera

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-06
"""

from aurora.srv.resources.image import Image


class Left(Image):
    uri = '/images/left'


class Right(Image):
    uri = '/images/right'

