#!/usr/bin/env python
"""Start Database Server 

This server supports Rest API.
TODO(kyon): add description.

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-06

Usage:
    $ python start_server.py
"""

from flask import Flask
from flask_restful import Api

from aurora.srv.resources import imu
from aurora.srv.resources import records
from aurora.srv.resources import stereo
from aurora.srv.resources import maps
from aurora.srv.resources import states

#from aurora.srv.resources import image
#from aurora.srv.resources import float_array


## Flask Application
app = Flask(__name__)
api = Api(app)


## Routes
def register_resource(res):
    records.RECORDS[res.uri] = None
    api.add_resource(res, res.uri)

register_resource(records.List)
register_resource(records.All)
register_resource(imu.Acc)
register_resource(imu.Gyro)
register_resource(imu.Inc)
register_resource(stereo.Left)
register_resource(stereo.Right)
register_resource(maps.VisualMap)
register_resource(maps.CostMap)
register_resource(maps.ClassMap)
register_resource(states.PoseVO)

# sample
#register_resource(image.Image)
#register_resource(float_array.FloatArray)


## Main scripts
if __name__ == '__main__':
    app.run(debug=True)

