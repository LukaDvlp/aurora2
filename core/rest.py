#!/usr/bin/env python
"""REST client helper

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-06
"""

import urlparse
import requests as req
import cv2
import time


HTTP_ROOT = 'http://localhost:5000/'

def put_image(path, im, ts=None):
    if ts is None: ts = time.time()
    im_data = cv2.imencode('.png', im)[1].tostring().encode('base64')
    req.put(urlparse.urljoin(HTTP_ROOT, path), data={'timestamp': ts, 'data': im_data})


def put_array(path, arr, ts=None):
    if ts is None: ts = time.time()
    arr_data = ','.join(str(a) for a in arr)
    req.put(urlparse.urljoin(HTTP_ROOT, path), data={'timestamp': ts, 'data': arr_data})


## Sample code
if __name__ == '__main__':
    import numpy as np

    im = np.zeros((300, 400), dtype=np.uint8)
    put_image('/sample/image', im)

    arr = np.array([1.0, 2.0, 3.0])
    put_array('/sample/float_array', arr, 1.0)


