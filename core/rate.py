#!/usr/bin/env python 
'''Rate adjustment

@author Kyohei Otsu <kyon@ac.jaxa.jp>
@date   2015-07-25
'''

import time

class Rate:
    def __init__(self, rate, name=''):
        self.name = name
        self.target_sec = 1.0 / rate
        self.prev_time = time.time()
        self.current_sec = 0.0

    def sleep(self):
        now = time.time()
        self.current_sec = now - self.prev_time
        time.sleep(max(0, self.target_sec - self.current_sec))
        self.prev_time = time.time()

    def report(self):
        print 'INFO(time): {} Exec={:.2f} ms  Target={:.2f} ms'.format(self.name, 1000 * self.current_sec, 1000 * self.target_sec)


