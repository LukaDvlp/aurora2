#!/usr/bin/env python
"""Records handler

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-05
"""

from flask_restful import Resource


## data table shared by every measurements
RECORDS = {}


class List(Resource):
    uri = '/'

    def get(self):
        return sorted(RECORDS.keys())


class All(Resource):
    uri = '/all'

    def get(self):
        return RECORDS
