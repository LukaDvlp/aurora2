#!/usr/bin/env python
"""Wrapper module for socket server

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26

Usage:
    $ python ***.py <args>
"""

import sys
import socket
import select
import signal

term_flag = False


class ServerBase:
    def __init__(self, sock):
        self.sock = sock


    def __del__(self):
        try:
            self.sock.shutdown(2)
            self.sock.close()
        except:
            pass
        print 'socket closed'
        

    def setup(self):
        ''' Perform initialization '''
        pass

    
    def worker(self):
        ''' Main procedure called in the every loop '''
        pass


    def handler(self, msg):
        ''' Message handler '''
        pass


    def finalize(self):
        ''' Destructor process '''
        pass



def start(serverinfo, MyServer):

    print 'INFO(comm): Starting socket server'

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(serverinfo)
    server.listen(1)
    server.settimeout(5)

    while True:
        if term_flag: break

        try:
            csock, caddr = server.accept()
        except:
            continue

        print 'INFO(comm): Accept from {}'.format(caddr)
        srv = MyServer(csock)

        srv.setup()
        while True:
            if term_flag: 
                break
            r, w, e = select.select([csock,], [csock,], [], 5)
            msg = ""
            if r:
                msg = csock.recv(1024)
                if len(msg) == 0: break  # socket closed
            srv.handler(msg)
            srv.worker()
        srv.finalize()
        del srv

    server.close()
    print 'Server terminated.....'


def signal_handler(signal, frame):
    global term_flag
    term_flag = True



if __name__ == '__main__':

    import sys
    import time
    
    socket_server = sys.modules[__name__]


    class MyServer(ServerBase):
        def setup(self):
            print 'initialize'

        def worker(self):
            print "1 2 3 4 5"
            time.sleep(1)

        def handler(self, msg):
            print "Received: ", msg

        def finalize(self):
            print 'finalize'


    socket_server.start(("localhost", 4647), MyServer)

