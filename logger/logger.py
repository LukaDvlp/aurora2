#!/usr/bin/env python
"""Logger module

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26

Usage:
    $ python logger.py
"""


import socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def setup():
    pass


def loop():
    pass


## Sample code
if __name__ == '__main__':
    server.bind(('localhost', 6969))
    server.listen(1)

    while True:
        try:
            csock, caddr = server.accept()
        except:
            continue

        print 'accepted from {}'.format(caddr)

        while True:
            print '1 2 3 4 6'
            time.sleep(1)

    raw_input()  # wait key input

