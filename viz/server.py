from flask import Flask, render_template, request
import cv2
import socket
import time
import threading
import numpy as np
from pymodbus.client.sync import ModbusTcpClient


###########################################################
#! flask etc.
app = Flask(__name__)
sock = None

#! modbus etc.
adc1 = ModbusTcpClient('192.168.201.17', port=502)
adc2 = ModbusTcpClient('192.168.201.18', port=502)
meas = np.zeros(16)


###########################################################
@app.route('/')
def show_main():
    return render_template('index.html')

@app.route('/send_goal', methods=['POST'])
def send_goal():
    #startUV = (float(request.form['startU']), float(request.form['startV']))
    goalUV  = (float(request.form['goalU']), float(request.form['goalV']))
    print goalUV
    sock.send('xy{:.2f},{:.2f}'.format(goalUV[0], goalUV[1]))
    return 'ok'

@app.route('/sensor/all')
def get_all_sensors():
    return ','.join(['{:.2f}'.format(m) for m in meas])

@app.route('/sensor/imu/roll')
def get_roll():
    return "{:.2f}".format(meas[0])

@app.route('/sensor/imu/pitch')
def get_pitch():
    return "{:.2f}".format(meas[1])

@app.route('/sensor/bus/com-busv')
def get_com_busv():
    return "{:.2f}".format(meas[1])

@app.route('/sensor/bus/mob-busv')
def get_mob_busv():
    return "{:.2f}".format(meas[1])

###########################################################
def updateMeasurement(rate):
    global meas

    def raw2volt(raw):
        return [10.0 * (m - 2**15) / 2**15 for m in raw]
    reg1 = adc1.read_input_registers(0, 8)
    reg2 = adc2.read_input_registers(0, 8)

    vlt1 = raw2volt(reg1.registers)
    vlt2 = raw2volt(reg2.registers)
    

    # AD Converter #1
    meas[ 0] = (vlt1[0] - 2.5185) / 1.2075  # AccX
    meas[ 1] = (vlt1[1] - 2.5172) / 1.2138  # AccY
    meas[ 2] = (vlt1[2] - 2.5195) / 1.2075  # AccZ
    meas[ 3] = np.arcsin((vlt1[3] - 2.5064) / 3.8134)  # IncX
    meas[ 4] = np.arcsin((vlt1[4] - 2.4962) / 3.9963)  # IncY
    meas[ 5] = (vlt1[5] - 2.4067) / 0.0799  # Gyro
    meas[ 6] = vlt1[6]
    meas[ 7] = vlt1[7]

    meas[ 8] = (vlt2[0] - 2.5) / 1.2  # AccX  ???
    meas[ 9] = (vlt2[1] - 2.5) / 1.2  # AccY  ???
    meas[10] = (vlt2[2] - 2.5) / 1.2  # AccZ  ???
    meas[11] = vlt2[3]
    meas[12] = vlt2[4] * 4.0  # BUS_V 28
    meas[13] = vlt2[5] * 4.0  # BUS_V 14
    meas[14] = (vlt2[6] - 2.5) / 0.037  # BUS_I 28
    meas[15] = (vlt2[7] - 2.5) / 0.037  # BUS_I 14
    #print meas
    threading.Timer(1.0 / rate, updateMeasurement, args=[rate]).start()

    
    

###########################################################
#! main
if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.connect(('192.168.201.10', 5557))
    #updateMeasurement(5)
    app.run(debug=True)
    #app.run(debug=True, host='192.168.201.10')
