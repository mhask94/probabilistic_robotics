#!/usr/bin/python3

import sys
import numpy as np
from scipy.io import loadmat

__usage__ = 'Usage: python3 occupancy_grid_mapping.py <filename>.mat'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == "__main__":
    args = sys.argv[1:]
    if sys.version_info[0] < 3:
        __error__('Requires Python 3.')

    if len(args) == 1:
        filename = args[0]
        if not filename[-4:] == '.mat':
            __error__('Invalid file extention, expected .mat')
        data = loadmat(filename)
    else:
        __error__('Invalid number of arguments, expected 1.\n' + __usage__)

    pose = data['X']
    thk = data['thk']
    z = data['z']

    for i in range(len(pose[0])):
        
