#!/usr/bin/env python
"""Logger module

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26

Usage:
    $ python logger.py
"""

import time

from aurora.core import server_wrapper
from aurora.core import rate
from aurora.viz import daemons


class LoggerServer(server_wrapper.ServerBase):
    def setup(self):
        self.r = rate.Rate(10, name="logger")
        self.cnt = 0

        self.adc = daemons.ADCDaemon(hz=10)
        self.dpc = daemons.DynPickDaemon(hz=10)
        self.cms = daemons.CompassDaemon(hz=1)

        self.adc.start()
        self.dpc.start()
        self.cms.start()

        pass


    def worker(self):
        #print time.time(), '1 2 3 4 5'
        #print views.adc_daemon.get_data()
        stamp = time.time()
        print get_time_formatted(stamp), self.adc.get_data()
        print get_time_formatted(stamp), self.dpc.get_data()
        print get_time_formatted(stamp), self.cms.get_data()
        self.r.sleep()
        self.cnt += 1


    def handler(self, msg):
        pass


    def finalize(self):
        print 'logger closing'
        self.adc.stop()
        pass


    def get_time_formatted(self, d):
        return d.strftime("%Y-%m-%dT%H:%M:%S.") + d.strftime("%f")[:3]

if __name__ == '__main__':

    server_wrapper.start(("localhost", 8888), LoggerServer)


