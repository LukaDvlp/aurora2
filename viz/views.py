#!/usr/bin/env python
"""View functions for Flask

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26
"""


import sys
import signal
import socket
import time
import threading

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


@app.route('/')
def show_main():
    return render_template('index.html')


# =============================================================== #


adc = [ModbusTcpClient('192.168.201.17', port=502),
       ModbusTcpClient('192.168.201.13', port=502)]
adc_status = [False, False]
adc_channels = 8
adc_meas = [np.zeros(adc_channels), np.zeros(adc_channels)]
adc_term_flag = False

def adc_daemon(hz):
    global adc_term_flag
    r = rate.Rate(hz, name='adc_daemon')
    while not term_flag and not adc_term_flag:
        update_adc()
        r.sleep()

    print 'ADC finish'
    adc_term_flag = False


def update_adc():
    global adc_meas

    def raw2volt(raw):
        return np.array([10.0 * (m - 2**15) / 2**15 for m in raw])

    # AD Converter #0
    idx = 0
    if adc_status[idx]:
        reg = adc[idx].read_input_registers(0, adc_channels)
        vlt = raw2volt(reg.registers)
        
        adc_meas[idx][0] = (vlt[0] - 2.5185) / 1.2075  # AccX
        adc_meas[idx][1] = (vlt[1] - 2.5172) / 1.2138  # AccY
        adc_meas[idx][2] = (vlt[2] - 2.5195) / 1.2075  # AccZ
        adc_meas[idx][3] = (vlt[3] - 2.4067) / 0.0799  # Gyro
        adc_meas[idx][4] = np.arcsin((vlt[4] - 2.5064) / 3.8134)  # IncX
        adc_meas[idx][5] = np.arcsin((vlt[5] - 2.4962) / 3.9963)  # IncY
        adc_meas[idx][6] = vlt[6]
        adc_meas[idx][7] = vlt[7]
    else:
        adc_meas[idx] = np.zeros(adc_channels)

    # AD Converter #1
    idx = 1
    if adc_status[idx]:
        reg = adc[idx].read_input_registers(0, adc_channels)
        vlt = raw2volt(reg.registers)

        adc_meas[idx][0] = (vlt[0] - 2.5) / 1.2  # AccX  ???
        adc_meas[idx][1] = (vlt[1] - 2.5) / 1.2  # AccY  ???
        adc_meas[idx][2] = (vlt[2] - 2.5) / 1.2  # AccZ  ???
        adc_meas[idx][3] = vlt[3]
        adc_meas[idx][4] = vlt[4] * 4.0  # BUS_V 28
        adc_meas[idx][5] = vlt[5] * 2.0  # BUS_V 14
        adc_meas[idx][6] = 0
        adc_meas[idx][7] = 0
        print vlt[0:3]
        #adc_meas[idx][6] = (vlt[6] - 2.5) / 0.037  # BUS_I 28
        #adc_meas[idx][7] = (vlt[7] - 2.5) / 0.037  # BUS_I 14
    else:
        adc_meas[idx] = np.zeros(adc_channels)

    #print ' '.join(['{:.2f}'.format(m) for m in adc_meas[0]] + ['{:.2f}'.format(m) for m in adc_meas[1]])
    print "==================="
    '''
    print "AccX: {:.2f}".format(adc_meas[0][0])
    print "AccY: {:.2f}".format(adc_meas[0][1])
    print "AccZ: {:.2f}".format(adc_meas[0][2])
    print "Gyro: {:.2f}".format(adc_meas[0][3])
    print "IncX: {:.2f}".format(adc_meas[0][4] / 3.14 * 180)
    print "IncY: {:.2f}".format(adc_meas[0][5] / 3.14 * 180)
    print "AccX: {:.2f}".format(adc_meas[1][0])
    print "AccY: {:.2f}".format(adc_meas[1][1])
    print "AccZ: {:.2f}".format(adc_meas[1][2])
    print "V_28: {:.2f}".format(adc_meas[1][4])
    print "V_14: {:.2f}".format(adc_meas[1][5])
    print "I_28: {:.2f}".format(adc_meas[1][6])
    print "I_14: {:.2f}".format(adc_meas[1][7])
    '''
    #threading.Timer(1.0 / rate, update_adc, args=[rate]).start()


@app.route('/adc/start')
def adc_start():
    global adc_term_flag, adc_status

    adc_term_flag = True
    adc_status[0] = adc[0].connect()
    adc_status[1] = adc[1].connect()
    print 'Modbus Connection: {}, {}'.format(adc_status[0], adc_status[1])
    adc_term_flag = False
    adc_thread = threading.Thread(target=adc_daemon, args=(1,))
    adc_thread.start()
    #update_adc(rate=1)
    return ''


@app.route('/adc/stop')
def adc_stop():
    global adc_term_flag
    adc_term_flag = True
    for i in range(len(adc_status)):
        if adc_status[i]:
            adc[i].close()
    return ''


@app.route('/adc/get_all')
def adc_get_all():
    return ' '.join(['{:.2f}'.format(m) for m in adc_meas[0]] + ['{:.2f}'.format(m) for m in adc_meas[1]])


# =============================================================== #

vision_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
vision_sock.setblocking(0)
vision_term_flag = False
vision_thread = None
vision_msg = "init"
vision_pose = np.zeros(3)


def vision_daemon(hz):
    global vision_term_flag, vision_sock
    global vision_pose
    r = rate.Rate(hz, name='vision_daemon')
    while not term_flag and not vision_term_flag:
        try:
            line = vision_sock.recv(1024)
            vision_pose = np.array([float(n) for n in line.split('\n')[0].split(' ')])
        except:
            pass
        r.sleep()

    print 'Vision finish'
    vision_term_flag = False


@app.route('/vision/start')
def vision_start():
    global vision_sock
    global vision_term_flag
    global vision_thread
    try:
        vision_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vision_sock.setblocking(0)
        vision_sock.connect(('localhost', 7777))
    except Exception, e:
        print 'Socket error: {}'.format(e)
        pass
    vision_term_flag = False
    vision_thread = threading.Thread(target=vision_daemon, args=(1,))
    vision_thread.start()
    return ''


@app.route('/vision/stop')
def vision_stop():
    global vision_term_flag
    vision_term_flag = True
    try:
        vision_sock.close()
    except Exception, e:
        print 'Socket error: {}'.format(e)
        pass
    return ''

@app.route('/vision/get_pose')
def vision_get_pose():
    global vision_pose
    return ' '.join(['{:.2f}'.format(m) for m in vision_pose])


@app.route('/vision/enable_drive', methods=['POST'])
def vision_enable_drive():
    status = float(request.form['status'])
    #sock.send('xy{:.2f},{:.2f}'.format(goalUV[0], goalUV[1]))
    return 'ok'

@app.route('/vision/set_goal', methods=['POST'])
def vision_set_goal():
    #startUV = (float(request.form['startU']), float(request.form['startV']))
    goalUV  = (float(request.form['goalU']), float(request.form['goalV']))
    print goalUV
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



