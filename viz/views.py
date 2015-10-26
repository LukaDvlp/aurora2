#!/usr/bin/env python
"""View functions for Flask

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26
"""

from flask import render_template, request
import socket
import time
import threading
import numpy as np
from pymodbus.client.sync import ModbusTcpClient

from aurora.viz import app


@app.route('/')
def show_main():
    return render_template('index.html')


# =============================================================== #


adc = [ModbusTcpClient('192.168.201.17', port=502),
       ModbusTcpClient('192.168.201.13', port=502)]
adc_status = [False, False]
adc_channels = 8
adc_meas = [np.zeros(adc_channels), np.zeros(adc_channels)]
adc_rate = 5
@app.route('/adc/start')
def adc_start():
    adc_status[0] = adc[0].connect()
    adc_status[1] = adc[1].connect()
    print 'Modbus Connection: {}, {}'.format(adc_status[0], adc_status[1])
    update_adc(adc_rate)
    return ''


def update_adc(rate):
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
        adc_meas[idx][3] = np.arcsin((vlt[3] - 2.5064) / 3.8134)  # IncX
        adc_meas[idx][4] = np.arcsin((vlt[4] - 2.4962) / 3.9963)  # IncY
        adc_meas[idx][5] = (vlt[5] - 2.4067) / 0.0799  # Gyro
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
        adc_meas[idx][6] = (vlt[6] - 2.5) / 0.037  # BUS_I 28
        adc_meas[idx][7] = (vlt[7] - 2.5) / 0.037  # BUS_I 14
    else:
        adc_meas[idx] = np.zeros(adc_channels)

    threading.Timer(1.0 / rate, update_adc, args=[rate]).start()


@app.route('/adc/get_all')
def adc_get_all():
    return ' '.join(['{:.2f}'.format(m) for m in adc_meas[0]] + ['{:.2f}'.format(m) for m in adc_meas[1]])


# =============================================================== #

vision_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
vision_sock.setblocking(0)

@app.route('/vision/start')
def vision_start():
    try:
        #vision_sock.connect(('192.168.201.10', 5557))
        vision_sock.connect(('localhost', 10000))
    except Exception, e:
        print 'Socket error: {}'.format(e)
        pass
    return ''


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



