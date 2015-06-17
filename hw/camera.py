#!/usr/bin/env python
"""Communication module to camera

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-06-12
"""

import threading
import Queue
import yaml

from skimage import io
import cv2

from aurora.core import core, decorator


# Frame number
FRAME = -1

# Config
CONFIG = {}


@decorator.runonce
def setup(yamlfile):
    '''load config from yaml'''
    global CONFIG
    data = open(yamlfile).read()
    CONFIG = yaml.load(data)
    if CONFIG.has_key('frame_range'):
        FRAME = CONFIG['frame_range'][0] - 1


def check_range():
    '''check if the frame range is valid

    Returns:
        True if the range is okay. 
    '''
    ok = not CONFIG.has_key('frame_range') or FRAME <= CONFIG['frame_range'][1]
    return ok


#@decorator.timeit
def get_raw_image(uri, q):
    '''Get raw image from uri

    Args:
        uri: URI for image
        q: Queue of fetched image
    '''
    try:
        im = io.imread(uri)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    except:
        im = None
    q.put(im)


def get_mono_image():
    '''Get monocular image

    Returns:
        frame: frame number
        imL: numpy array in BGR order (OpenCV style)
        imR: numpy array in BGR order (OpenCV style)
    '''
    # update frame number
    global FRAME
    FRAME = FRAME + 1

    # generate URI
    if check_range():
        uriL = CONFIG['cameraL']['uri'].format(FRAME)
    else:
        uriL = ''

    # get image
    qimL = Queue.Queue()
    get_raw_image(uriL, qimL)

    try:
        imL = qimL.get_nowait()
    except:
        imL = None

    return FRAME, imL


def get_stereo_images():
    '''Get stereo images

    Returns:
        frame: frame number
        imL: numpy array in BGR order (OpenCV style)
        imR: numpy array in BGR order (OpenCV style)
    '''
    # update frame number
    global FRAME
    FRAME = FRAME + 1

    # generate URI
    if check_range():
        uriL = CONFIG['cameraL']['uri'].format(FRAME)
        uriR = CONFIG['cameraR']['uri'].format(FRAME)
    else:
        uriL, uriR = '', ''

    # get image
    qimL, qimR = Queue.Queue(), Queue.Queue()
    threads = (threading.Thread(target=get_raw_image, args=(uriL, qimL)), 
               threading.Thread(target=get_raw_image, args=(uriR, qimR)))
    for t in threads: t.start()
    for t in threads: t.join()

    try:
        imL = qimL.get_nowait()
        imR = qimR.get_nowait()
    except:
        imL, imR = None, None

    return FRAME, imL, imR


## Sample code
if __name__ == '__main__':
    setup(core.get_full_path('config/camera_config.yaml'))

    frame, imL = get_mono_image()
    frame, imL, imR = get_stereo_images()

    cv2.imshow('image', imL)
    cv2.waitKey(0)

