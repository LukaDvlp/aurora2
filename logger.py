#!/usr/bin/env python
"""Logger module

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26

Usage:
    $ python logger.py
"""

import time
import logging
import logging.config
import yaml
from aurora.core import core

from aurora.core import server_wrapper
from aurora.core import rate
from aurora.viz import daemons


class LoggerServer(server_wrapper.ServerBase):
    def setup(self):
        self.r = rate.Rate(10, name="logger")
        self.cnt = 0

        adc_yaml = core.get_full_path('config/logger/adc_logger.yaml')
        data = open(adc_yaml).read()
        logging.config.dictConfig(yaml.load(data))
        self.log_adc = logging.getLogger('logger_adc')
        self.log_dpc = logging.getLogger('logger_dynpick')
        self.log_cms = logging.getLogger('logger_compass')

        #self.adc = daemons.ADCDaemon(hz=10)
        #self.dpc = daemons.DynPickDaemon(hz=10)
        self.cms = daemons.CompassDaemon(hz=1)

        self.adc.start()
        #self.dpc.start()
        self.cms.start()


        pass


    def worker(self):
        #print time.time(), '1 2 3 4 5'
        #print views.adc_daemon.get_data()
        stamp = time.time()
        #self.log_adc.info(self.adc.get_data())
        #self.log_adc.info(self.dpc.get_data())
        self.log_cms.info(self.cms.get_data())
        self.r.sleep()
        self.cnt += 1


    def handler(self, msg):
        pass


    def finalize(self):
        print 'logger closing'
        #self.adc.stop()
        #self.dpc.stop()
        self.cms.stop()
        pass


    def get_time_formatted(self, d):
        return d.strftime("%Y-%m-%dT%H:%M:%S.") + d.strftime("%f")[:3]

if __name__ == '__main__':

    server_wrapper.start(("localhost", 8888), LoggerServer)


