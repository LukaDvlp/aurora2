#!/usr/bin/env python
"""Driver for AD Converter 

@author  Kyohei Otsu <kyon@ac.jaxa.jp>
@date    2015-10-23

Usage:
    $ python adc.py
"""

import numpy as np
from pymodbus.client.sync import ModbusTcpClient

from aurora.core import decorator
from aurora.core import globval


# container for adc modules
mods = []


@decorator.runonce
def setup():
    ''' Initialize this module '''
    connect()


def connect():
    global mods
    mods = [0, 0]
    mods[0] = ModbusTcpClient('192.168.201.17', port=502)
    mods[1] = ModbusTcpClient('192.168.201.17', port=502)


def read():
    for i, mod in enumerate(mods):
        mod.read_input_registers()






## Sample code
if __name__ == '__main__':


    raw_input()  # wait key input

