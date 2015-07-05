#!/usr/bin/env python
"""Resource definition for maps

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-06
"""

from aurora.srv.resources.image import Image


class VisualMap(Image):
    uri = '/images/visual_map'


class CostMap(Image):
    uri = '/images/cost_map'


class ClassMap(Image):
    uri = '/images/class_map'

