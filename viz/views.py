#!/usr/bin/env python
"""View functions for Flask

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26
"""


import sys
import signal
import socket
import itertools
import time
import threading
import Queue

import numpy as np
from flask import render_template, request
from pymodbus.client.sync import ModbusTcpClient

from aurora.viz import app
from aurora.core import rate

# =============================================================== #
#  Signal Handler

term_flag = False

def signal_handler(signal, frame):
    print 'Terminating.....'
    global term_flag
    term_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

# =============================================================== #
#   Message Handler

class Messenger:
    def __init__(self):
        self.msgs = Queue.Queue()


    def set(self, msg):
        self.msgs.put(msg)


    def get(self):
        msg = ""
        while not self.msgs.empty():
            msg += self.msgs.get_nowait() + "\n"
        return msg


msger = Messenger()


@app.route('/message/get')
def message_get():
    return msger.get()



# =============================================================== #

class DaemonBase(threading.Thread):
    def __init__(self, hz, name=""):
        threading.Thread.__init__(self)
        self.name = name
        self.rate = rate.Rate(hz, name=self.name)


    def run(self):
        self.setup()
        self.term_flag = False
        while not term_flag and not self.term_flag:
            self.worker()
            self.rate.sleep()
        print 'worker close ({})'.format(self.name)
        self.term_flag = False
        self.finalize()


    def stop(self):
        self.term_flag = True


    def set_msg(self, msg):
        msger.set("[{}] {}".format(self.name, msg))


    def setup(self):
        pass


    def worker(self):
        pass
    
    
    def finalize(self):
        pass


# =============================================================== #

class ADCDaemon(DaemonBase):
    def __init__(self, hz, name="adc"):
        DaemonBase.__init__(self, hz, name)
        self.adc = [ModbusTcpClient('192.168.201.17', port=502),
                    ModbusTcpClient('192.168.201.13', port=502)]
        self.adc_channels = 8
        self.data = np.zeros((2, self.adc_channels))


    def setup(self):
        self.status = [a.connect() for a in self.adc]
        self.set_msg('ADC Modbus connection: {}, {}'.format(self.status[0], self.status[1]))
        self.data = np.zeros((2, self.adc_channels))


    def worker(self):
        idx = 0
        if self.status[idx]:
            vlt = read(idx)
            self.data[idx][0] = (vlt[0] - 2.5185) / 1.2075  # AccX
            self.data[idx][1] = (vlt[1] - 2.5172) / 1.2138  # AccY
            self.data[idx][2] = (vlt[2] - 2.5195) / 1.2075  # AccZ
            self.data[idx][3] = (vlt[3] - 2.4067) / 0.0799  # Gyro
            self.data[idx][4] = np.arcsin((vlt[4] - 2.5064) / 3.8134)  # IncX
            self.data[idx][5] = np.arcsin((vlt[5] - 2.4962) / 3.9963)  # IncY
            self.data[idx][6] = vlt[6]
            self.data[idx][7] = vlt[7]

        idx = 1
        if self.status[idx]:
            vlt = read(idx)
            self.data[idx][0] = (vlt[0] - 2.5) / 1.2  # AccX  ???
            self.data[idx][1] = (vlt[1] - 2.5) / 1.2  # AccY  ???
            self.data[idx][2] = (vlt[2] - 2.5) / 1.2  # AccZ  ???
            self.data[idx][3] = vlt[3]
            self.data[idx][4] = vlt[4] * 4.0  # BUS_V 28
            self.data[idx][5] = vlt[5] * 2.0  # BUS_V 14
            self.data[idx][6] = (vlt[6] - 2.5) / 0.037  # BUS_I 28
            self.data[idx][7] = (vlt[7] - 2.5) / 0.037  # BUS_I 14


    def finalize(self):
        self.data = [np.zeros(self.adc_channels), np.zeros(self.adc_channels)]
        map(lambda a: a.close(), itertools.compress(self.adc, self.status))
        self.set_msg("ADC Modbus closed")


    def read(self, idx):
        data = np.zeros(self.adc_channels)
        if self.status[idx]:
            reg = self.adc[idx].read_input_registers(0, self.adc_channels)
            data = raw2volt(reg.registers)
        return data
    

    def get_data(self):
        return ' '.join(['{:.2f}'.format(m) for m in self.data.ravel()])


    def raw2volt(raw):
        return np.array([10.0 * (m - 2**15) / 2**15 for m in raw])


adc_daemon = ADCDaemon(hz=1)

@app.route('/adc/start')
def adc_start():
    global adc_daemon
    adc_daemon = ADCDaemon(hz=1, name="adc")
    adc_daemon.start()
    return ""


@app.route('/adc/stop')
def adc_stop():
    adc_daemon.stop()
    return ""


@app.route('/adc/get_all')
def adc_get_all():
    return adc_daemon.get_data()


# =============================================================== #


class VisionDaemon(DaemonBase):
    def __init__(self, hz, name="vision"):
        DaemonBase.__init__(self, hz, name)
        self.pose = np.zeros(3)
        self.sendq = Queue.Queue()

    def setup(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        ex = self.sock.connect_ex(("localhost", 7777))
        self.status = (ex == 0)
        self.set_msg('Socket open: {}'.format(self.status))
        self.pose = np.zeros(3)


    def worker(self):
        if self.status:
            # recv
            line = self.sock.recv(1024)
            arr = line.split('\n')[0].split(' ')
            if arr[0] == 'xyh':
                self.pose = np.array([float(v) for v in arr[1:]])
                print self.pose

            # send
            while not self.sendq.empty():
                msg = self.sendq.get()
                self.sock.send(msg + "\n")
                #time.time(0.01)
    
    
    def finalize(self):
        self.pose = np.zeros(3)
        self.sock.close()
        self.set_msg('socket closed')


    def get_pose(self):
        return ' '.join(['{:.2f}'.format(p) for p in self.pose])


    def set_drive(self, flag):
        if flag == True:
            self.sendq.put("d")
        else:
            self.sendq.put("f")


    def set_goal(self, goal):
        ''' The goal is in the pixel coordinates '''
        self.sendq.put("g {:2f} {:2f}".format(goal[0], goal[1]))


    def clear_goal(self):
        self.sendq.put("c")




vision_daemon = VisionDaemon(hz=1)

@app.route('/vision/start')
def vision_start():
    global vision_daemon
    vision_daemon = VisionDaemon(hz=1)
    vision_daemon.start()
    return ""


@app.route('/vision/stop')
def vision_stop():
    vision_daemon.stop()
    return ""


@app.route('/vision/pose/get')
def vision_pose_get():
    return vision_daemon.get_pose()


@app.route('/vision/goal/set', methods=['POST'])
def vision_goal_set():
    goalUV  = (float(request.form['goalU']), float(request.form['goalV']))
    vision_daemon.set_goal((goalUV[0], goalUV[1]))
    return ''

@app.route('/vision/goal/clear')
def vision_goal_clear():
    vision_daemon.clear_goal()
    return ''


@app.route('/drive/start')
def drive_start():
    vision_daemon.set_drive(True)
    return ""


@app.route('/drive/stop')
def drive_stop():
    vision_daemon.set_drive(False)
    return ""




@app.route('/vision/enable_drive', methods=['POST'])
def vision_enable_drive():
    status = float(request.form['status'])
    #sock.send('xy{:.2f},{:.2f}'.format(goalUV[0], goalUV[1]))
    return 'ok'

@app.route('/send_goal', methods=['POST'])
def send_goal():
    #startUV = (float(request.form['startU']), float(request.form['startV']))
    goalUV  = (float(request.form['goalU']), float(request.form['goalV']))
    print goalUV
    #sock.send('xy{:.2f},{:.2f}'.format(goalUV[0], goalUV[1]))
    return 'ok'


# =============================================================== #


logger_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
logger_sock.setblocking(0)

@app.route('/logger/start')
def logger_start():
    try:
        #logger_sock.connect(('192.168.201.10', 5557))
        logger_sock.connect(('localhost', 10000))
    except Exception, e:
        print 'Socket error: {}'.format(e)
        pass
    return ''



@app.route('/')
def show_main():
    return render_template('index.html')

