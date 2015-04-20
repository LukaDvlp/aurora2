#!/usr/bin/env python
""" Terrain classification using an image

Input:
    - RGB rectified image

Output:
    - Terrain class image (single channel, 0-255)

Written by Kyohei Otsu <kyon@ac.jaxa.jp> in 2015-04-18
"""

def classify(img, algorithm='sand_rock'):
    '''
        main function
    '''
    if algorithm == 'sand_rock':
        label = classify_sand_rock(img)
    elif algorithm == 'grass_rock':
        label = classify_grass_rock(img)
    else
        print 'Invalid algorithm: ' + algorithm
    return label


def classify_sand_rock(img):
    pass

def classify_grass_rock(img):
    pass



####################################
#  sample code                     #
####################################
if __name__ == '__main__':


    raw_input()  # wait key input


