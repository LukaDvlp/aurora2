#!/usr/bin/env python
"""View functions for Flask

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-26
"""


from flask import render_template, request

from aurora.viz import app
from aurora.viz import daemons

@app.route('/message/get')
def message_get():
    return daemons.msger.get()


adc_daemon    = daemons.ADCDaemon(hz=1)
vision_daemon = daemons.VisionDaemon(hz=1)
logger_daemon = daemons.LoggerDaemon(hz=1)
compass_daemon    = daemons.CompassDaemon(hz=1)

@app.route('/adc/start')
def adc_start():
    global adc_daemon
    adc_daemon = daemons.ADCDaemon(hz=1, name="adc")
    adc_daemon.start()
    return ""


@app.route('/adc/stop')
def adc_stop():
    adc_daemon.stop()
    return ""


@app.route('/adc/get_all')
def adc_get_all():
    return adc_daemon.get_data()


@app.route('/compass/start')
def compass_start():
    global compass_daemon
    compass_daemon = daemons.CompassDaemon(hz=1)
    compass_daemon.start()
    return ""


@app.route('/compass/stop')
def compass_stop():
    compass_daemon.stop()
    return ""


@app.route('/compass/get_all')
def compass_get_all():
    return compass_daemon.get_data()


@app.route('/vision/start')
def vision_start():
    global vision_daemon
    vision_daemon = daemons.VisionDaemon(hz=1)
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



@app.route('/logger/start')
def logger_start():
    global logger_daemon
    logger_daemon = daemons.LoggerDaemon(hz=0.1)
    logger_daemon.start()
    return ""


@app.route('/logger/stop')
def logger_stop():
    logger_daemon.stop()
    return ""


# =============================================================== #


@app.route('/')
def show_main():
    return render_template('index.html')

