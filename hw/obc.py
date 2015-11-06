'''
    Communication with Controller OBC
'''

import socket
import struct
import time

import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

sock = None
cmd_seq = 0

flag_turn = False
steer_angle = 0

def setup():
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    #sock.connect(("127.0.0.1", 5555))
    try:
    	sock.connect(("192.168.201.11", 13000))
    except:
        pass


def send_cmd(cmd_list):
    for c in cmd_list:
        logger.info('Commands Sent: {}'.format(c))
        try:
            sock.sendall(capsulate(c))
	    pass
        except:
            logger.error('  *** Command failed ***'.format(c))
            pass


def serialize(hexstr):
    ''' Convert hex string into byte lists '''
    assert len(hexstr) % 2 == 0

    serstr = ''
    for h in [hexstr[i:i+2] for i in range(0, len(hexstr), 2)]:
        serstr += chr(int(h, base=16))
    return serstr


def capsulate(cmd):
    ''' Append header and convert to byte strings'''
    global cmd_seq
    cmd_seq += 1

    marker = '0020f3fa00000000'
    sender_id = '0010'
    seq = '{:04d}'.format(cmd_seq)
    cmd_type = cmd[0]
    packet = None
    if len(cmd) > 1:
        cmd_arg = struct.pack('f', float(cmd[1:]))
    else:
        cmd_arg = struct.pack('f', float(0))
    packet = serialize(marker + sender_id + seq) + cmd_type + cmd_arg
    #print(marker + sender_id + seq) + cmd_type + cmd_arg
    return packet


def set_turn_mode(flag):
    '''
        enter/exit inspot turn mode. Warn: this function is blocking
    '''
    global flag_turn
    cmd_list = []
    timeout = 0
    if flag and not flag_turn:
        cmd_list.append('i')
        flag_turn = True
        timeout = 8
    elif not flag and flag_turn:
        cmd_list.append('f')
        flag_turn = False
        timeout = 8
    send_cmd(cmd_list)
    if timeout > 0:
        print 'INFO(control): Wait {} secs for steering'.format(timeout)
        time.sleep(timeout)  # wait for preparation


def set_steer_angle(angle):
    '''
        set steer angle. this function is blocking
    '''
    global steer_angle 
    cmd_list = []
    if abs(angle) < 0.1: angle = 0
    if angle > 20: angle = 20
    if angle < -20: angle = -20
    #cmd_list.append('s{:.2f}'.format(angle))
    cmd_list.append('u{:.2f}'.format(angle))
    print cmd_list
    timeout = 0.5
    if abs(steer_angle - angle) > 15:
        timeout = 2
    if abs(steer_angle - angle) > 0.1:
        steer_angle = angle
        send_cmd(cmd_list)
        if timeout > 0:
            print 'INFO(control): Wait {} secs for steering'.format(timeout)
            time.sleep(timeout)  # wait for preparation
    

