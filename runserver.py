#!/usr/bin/env python
"""Run Flask server

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26

Usage:
    $ python runserver.py (<ipaddr>)
"""

from aurora.viz import app


if __name__ == '__main__':
    app.run(debug=True)

