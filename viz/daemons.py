#!/usr/bin/env python
"""The summary of this module

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-11-02

Usage:
    $ python ***.py <args>
"""

import serial
import signal
import socket
import sys
import itertools
import threading
import Queue

import numpy as np
from pymodbus.client.sync import ModbusTcpClient

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
        self.adc = [ModbusTcpClient('192.168.201.21', port=502),
                    ModbusTcpClient('192.168.201.22', port=502)]
        self.adc_channels = 8
        self.data = np.zeros((2, self.adc_channels))


    def setup(self):
        self.status = [a.connect() for a in self.adc]
        self.set_msg('ADC Modbus connection: {}, {}'.format(self.status[0], self.status[1]))
        self.data = np.zeros((2, self.adc_channels))


    def worker(self):
        idx = 0
        if self.status[idx]:
            vlt = self.read(idx)
            self.data[idx][0] = (vlt[0] - 2.5185) / 1.2075  * 9.8067 # AccX
            self.data[idx][1] = (vlt[1] - 2.5172) / 1.2138  * 9.8067 # AccY
            self.data[idx][2] = (vlt[2] - 2.5195) / 1.2075  * 9.8067 # AccZ
            self.data[idx][3] = (vlt[3] - 2.4067) / 0.0799  * 9.8067 # Gyro
            self.data[idx][4] = np.arcsin((vlt[4] - 2.5064) / 3.8134)  # IncX
            self.data[idx][5] = np.arcsin((vlt[5] - 2.4962) / 3.9963)  # IncY
            self.data[idx][6] = vlt[6]
            self.data[idx][7] = vlt[7]

        idx = 1
        if self.status[idx]:
            vlt = self.read(idx)
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
            data = self.raw2volt(reg.registers)
        return data
    

    def get_data(self):
        return ' '.join(['{:.6f}'.format(m) for m in self.data.ravel()])


    def raw2volt(self, raw):
        return np.array([10.0 * (m - 2**15) / 2**15 for m in raw])


# =============================================================== #


class DynPickDaemon(DaemonBase):
    def __init__(self, hz, name="dynpick"):
        DaemonBase.__init__(self, hz, name)
        self.dpc = [serial.Serial('/dev/ttyS8', baudrate=921600, timeout=5),
                    serial.Serial('/dev/ttyS9', baudrate=921600, timeout=5),
                    serial.Serial('/dev/ttyS4', baudrate=921600, timeout=5),
                    serial.Serial('/dev/ttyS5', baudrate=921600, timeout=5)]
        self.status = [False] * len(self.dpc)
        self.dpc_channels = 6
        self.data = np.zeros((len(self.dpc), self.dpc_channels))

        self.gain = np.array([
            [1/32.6, 1/32.8, 1/32.7, 1/1627.3, 1/1617.6, 1/1618.6],  #3
            [1/32.6, 1/32.9, 1/32.6, 1/1611.5, 1/1610.9, 1/1643.9],  #2 
            [1/32.6, 1/32.7, 1/32.6, 1/1592.4, 1/1619.1, 1/1642.5],  #4
            [1/32.7, 1/32.8, 1/32.5, 1/1612.0, 1/1627.0, 1/1619.1],  #1
            ])
        self.offset = 8192 * np.ones(self.gain.shape)


    def setup(self):
        for i in range(len(self.dpc)):
            if not self.dpc[i].isOpen():
                self.dpc[i].open()
            self.status[i] = self.dpc[i].isOpen()
        self.set_msg('DynPick connection: {}, {}, {}, {}'.format(self.status[0], self.status[1], self.status[2], self.status[3]))
        self.data = np.zeros((len(self.dpc), self.dpc_channels))


    def worker(self):
        for idx in range(len(self.dpc)):
            if self.status[idx]:
                meas = self.read(idx)
                self.data[idx] = self.gain[idx] * (meas - self.offset[idx])


    def finalize(self):
        self.data = np.zeros((len(self.dpc), self.dpc_channels))
        map(lambda a: a.close(), itertools.compress(self.dpc, self.status))
        self.set_msg("DynPick serial closed")


    def read(self, idx):
        data = np.zeros(self.dpc_channels)
        if self.status[idx]:
            #self.dpc[idx].write("R")
            res = self.dpc[idx].readline()
        return np.array([int(res[ 1: 5], 16), int(res[ 5: 9], 16), int(res[ 9:13], 16),
                         int(res[13:17], 16), int(res[17:21], 16), int(res[21:25], 16)])
    

    def get_data(self):
        return ' '.join(['{:.6f}'.format(m) for m in self.data.ravel()])


# =============================================================== #


class CompassDaemon(DaemonBase):
    def __init__(self, hz, name="compass"):
        DaemonBase.__init__(self, hz, name)
        self.compass = serial.Serial('/dev/ttyS6', baudrate=19200, timeout=5)
        self.status = self.compass.isOpen()
        self.channels = 3
        self.data = np.zeros(self.channels)
        self.buf = ''


    def setup(self):
        if not self.compass.isOpen():
            self.compass.open()
            self.status = self.compass.isOpen()
        self.set_msg('Compass connection: {}'.format(self.status))
        self.data = np.zeros(self.channels)


    def worker(self):
        self.data = self.read()


    def finalize(self):
        self.data = np.zeros(self.channels)
        self.compass.close()
        self.status = self.compass.isOpen()
        self.set_msg("Compass serial closed")


    def read(self):
        if self.status:
            self.buf += self.compass.read(self.compass.inWaiting())
            if '\n' in self.buf:
                line, self.buf = self.buf.split('\n')[-2:]
                a = line.split(',')
                if len(a) > 0 and a[0] == '$PTNTHPR':
                    try:
                        h = float(l[1])
                        status_h = l[2]
                        if status_h != 'N':
                            self.data[0] = h
                        pass
                    except:
                        pass
    

    def get_data(self):
        return ' '.join(['{:.6f}'.format(m) for m in self.data.ravel()])


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
        return ' '.join(['{:.6f}'.format(p) for p in self.pose])


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

# =============================================================== #

class LoggerDaemon(DaemonBase):
    def __init__(self, hz, name="logger"):
        DaemonBase.__init__(self, hz, name)
        self.sendq = Queue.Queue()

    def setup(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        ex = self.sock.connect_ex(("localhost", 8888))
        self.status = (ex == 0)
        self.set_msg('Socket open: {}'.format(self.status))


    def worker(self):
        pass
    
    
    def finalize(self):
        self.sock.close()
        self.set_msg('socket closed')

