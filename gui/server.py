#!/usr/bin/env python
""" Web server module for GUI

Usage:
    $ python server.py

Written by Kyohei Otsu <kyon@ac.jaxa.jp> since 2015-04-20
"""

import time
import os

import numpy as np
from flask import Flask, render_template, request
import cv2

import aurora

app = Flask(__name__)
planner = aurora.planning.Planner((500, 500))

DATADIR = '/Users/kyoheee/Codes/aurora/gui/static/data'

@app.route("/")
def showMain():
    return render_template('index.html')

@app.route("/planning", methods=['POST'])
def makePath():
    startU = int(float(request.form['startU']))
    startV = int(float(request.form['startV']))
    goalU = int(float(request.form['goalU']))
    goalV = int(float(request.form['goalV']))

    tmap = cv2.imread(os.path.join(DATADIR, 'cmap.png'))
    start_time = time.time()
    waypoints, score = planner.plan_path(tmap, (startV, startU), (goalV, goalU));
    end_time = time.time()
    if waypoints is None: raise
    waypoints = (250 - waypoints) * 0.02
    np.savetxt(os.path.join(DATADIR, 'path.csv'), waypoints, delimiter=',')
    return "Path score: %s <br/> Execution Time: %.2f s" %(score, end_time - start_time)


if __name__ == '__main__':
    app.run(debug=True)

