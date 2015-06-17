#!/usr/bin/env python
"""Top-level module of aurora rover controller

@author kyohei otsu <kyon@ac.jaxa.jp>
@date   2015-06-12

Usage:
    $ python start.py
"""

import threading
import Queue
import flask

import numpy as np
import cv2

from aurora.core import core
from aurora.hw import camera
from aurora.loc import libviso2 as vo


## Global variables

# Main application
APP = flask.Flask('viz')
IPADDR = '127.0.0.1'

# Goals
GOALS = Queue.Queue()


## Functions for image processing

def setup():
    '''Execute initialization procedure

    This method should not block.
    '''
    camera.setup(core.get_full_path('config/camera_local_config.yaml'))

    vo_param = np.array([487.76, 320., 240, 0.105])
    vo.setup(vo_param)


def loop():
    '''Main loop for processing new images'''
    goal = None
    while True:
        # get image
        frame, imL, imR = camera.get_stereo_images()
        if imL is None: break
        imLg = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imRg = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

        # update pose
        wTc = vo.update_stereo(imLg, imRg)
        print frame, wTc[:3, 3].T

        # update map

        # fetch new goal from queue
        if goal is None:
            try: 
                goal = GOALS.get_nowait()
            except: 
                pass

        # goal reached?

        if goal is not None:
            # detect obstacles & update hazard map
            # generate waypoints
            # send commands
            pass


def set_new_goal(new_goal, clear_all=False):
    '''Set new goal to queue

    Args:
        new_goal: A tuple that contains (X, Y) in global coordinates.
        clear_all: If set, all previous goals are cleared.
    '''
    if clear_all:
        with GOALS.mutex: GOALS.queue.clear()
    GOALS.put(new_goal)


## Flask URI binding

@APP.route('/')
def uri_root():
    '''Render main page'''
    return flask.render_template('top.html')


@APP.route('/set/goal', methods=['POST'])
def uri_set_goal():
    '''Set goal from web UI

    The goal should be specified in global coordinates
    '''
    goal = float(request.form['goalX']), float(request.form['goalY'])
    set_new_goal(goal, clear_all=True)


if __name__ == '__main__':
    setup()
    thread = threading.Thread(target=loop)
    thread.setDaemon(True)
    thread.start()
    APP.run(host=IPADDR, debug=False)

