#!/usr/bin/env python
"""Resource definition for image

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-05
"""

from flask import send_file
from flask_restful import Resource, reqparse, abort
from aurora.srv.resources import records
from aurora.core import core


## Argument parser
parser = reqparse.RequestParser()
parser.add_argument('timestamp')
parser.add_argument('data')


## Resource definition
class Image(Resource):
    uri  = '/sample/image'

    def get(self):
        self.abort_if_not_exist(self.uri)
        return send_file(records.RECORDS[self.uri]['data'], mimetype='image/png')


    def put(self):
        args = parser.parse_args()
        records.RECORDS[self.uri] = {
                'timestamp': float(args['timestamp']),
                'data':      core.get_full_path('{}.png'.format(self.uri.replace('/', '_'))),
                }
        f = open(records.RECORDS[self.uri]['data'], 'w')
        f.write(args['data'].decode('base64'))
        f.close()
        return records.RECORDS[self.uri]


    def abort_if_not_exist(self, uri):
        if uri not in records.RECORDS:
            abort(404, message="Field {} doesn't exist".format(uri))
        if records.RECORDS[uri] is None:
            abort(404, message="Data {} doesn't exist".format(uri))



## Sample code
if __name__ == '__main__':
    from requests import put, get
    import numpy as np
    import cv2

    # create dummy image
    im = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.circle(im, (150, 200), 100, (0, 255, 0), -1)

    # encode image
    im_data = cv2.imencode('.png', im)[1].tostring().encode('base64')
    
    print 'Upload an image'
    print put('http://localhost:5000/sample/image', data={'data':im_data, 'timestamp':1.3})

    print 'Download an image'
    print get('http://localhost:5000/sample/image')


