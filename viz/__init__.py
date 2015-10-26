#!/usr/bin/env python
"""Main server

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26
"""

import flask

app = flask.Flask(__name__)

import aurora.viz.views

