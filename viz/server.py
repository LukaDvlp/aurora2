from flask import Flask, render_template, request
import cv2
import socket
import time
import threading
from pymodbus.client.sync import ModbusTcpClient


###########################################################
#! flask etc.
app = Flask(__name__)
sock = None

#! modbus etc.
adc = ModbusTcpClient('192.168.201.17', port=502)
meas = [0] * 8


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

    def raw2volt(meas):
        return [10.0 * (m - 2**15) / 2**15 for m in meas]
    rr = adc.read_input_registers(0, 8)
    meas = raw2volt(rr.registers)
    #print meas
    threading.Timer(1.0 / rate, updateMeasurement, args=[rate]).start()

    
    

###########################################################
#! main
if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.connect(('192.168.201.10', 5557))
    updateMeasurement(10)
    app.run(debug=True)
    #app.run(debug=True, host='192.168.201.10')
