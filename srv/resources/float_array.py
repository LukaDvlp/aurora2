#!/usr/bin/env python
"""Resource definition for float array

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-05
"""

from flask_restful import Resource, reqparse, abort
from aurora.srv.resources import records


## Argument parser
parser = reqparse.RequestParser()
parser.add_argument('timestamp')
parser.add_argument('data')


class FloatArray(Resource):
    uri  = '/sample/float_array'

    def get(self):
        self.abort_if_not_exist(self.uri)
        return records.RECORDS[self.uri]


    def put(self):
        args = parser.parse_args()
        records.RECORDS[self.uri] = {
                'timestamp': float(args['timestamp']),
                'data':      [float(v) for v in args['data'].split(",")],
                }
        return records.RECORDS[self.uri]


    def abort_if_not_exist(self, uri):
        if uri not in records.RECORDS:
            abort(404, message="Field {} doesn't exist".format(uri))
        if records.RECORDS[uri] is None:
            abort(404, message="Data {} doesn't exist".format(uri))


## Sample code
if __name__ == '__main__':
    from requests import put, get

    # sample data
    data = [3.3, 2.2, 1.1]
    data_str = ','.join(map(str, data))


    print 'Upload an array'
    print put('http://localhost:5000/sample/float_array', data={'data':data_str, 'timestamp':1.3}).json()

    print 'Download an array'
    print get('http://localhost:5000/sample/float_array').json()


