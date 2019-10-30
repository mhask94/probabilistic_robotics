#!/usr/bin/python3

import sys
import numpy as np
from visualizer import Visualizer
from scipy.io import loadmat
from quadcopter import Quadcopter
from extended_info_filter import ExtendedInfoFilter
from utils import wrap

__usage__ = '\nUsage:\tpython3 main.py <filename>.mat'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == "__main__":
    args = sys.argv[1:]
    if sys.version_info[0] < 3:
        __error__('Requires Python 3' + __usage__)

    # load data from .mat file
    if len(args) == 1:
        mat_file = args[0]
        mat = loadmat(mat_file)
        time = mat['t'][0]
        ts, tf = time[1], time[-1]
        pose_tr, b_tr, r_tr = mat['X_tr'], mat['bearing_tr'], mat['range_tr']
        w, v = mat['om'][0], mat['v'][0]
        w_c, v_c = mat['om_c'][0], mat['v_c'][0]
        landmarks = mat['m'].T
        x0 = pose_tr[:,:1] 
    else:              
        print('Invalid number of arguments.' + __usage__)

    ## parameters
    M = np.diag([0.15, 0.1])**2 # process noise (control space)
    Q = np.diag([0.2, 0.1])**2  # sensor noise
    xhat0 = np.array([[0., 0., 0.]]).T
    sigma0 = np.diag([1., 1., 0.1])

    ## system (works great, but don't need because data was provided)
#    quadcopter = Quadcopter(Q, M, ts, x0, landmarks)
    
    ## extended information filter
    eif = ExtendedInfoFilter(Q, M, ts, xhat0, sigma0, landmarks)
    
    # plotting
    lims=[-30, 30, -30, 30]
    viz = Visualizer(lims, x0, xhat0, sigma0, landmarks, live=True)
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0: # already know initial conditions
            continue

        # input commands
        u_c = np.array([[v_c[i], w_c[i]]]).T
        u = np.array([[v[i], w[i]]]).T
    
        # propagate actual system
#        x1 = quadcopter.propagateDynamics(u) # exactly the same as pose_tr
        x1 = wrap(pose_tr[:,i:i+1], dim=2)

        # sensor measurement
#        z = quadcopter.getSensorMeasurement() # will give custom noise
        z = np.block([[r_tr[i]],[wrap(b_tr[i])]])
    
        # filter  
        xhat_bar = eif.predictionStep(u_c)
        info_vec, xhat, covariance, zhat = eif.correctionStep(z)
    
        # store plotting variables
        viz.update(t, x1, xhat, covariance, zhat, info_vec)
    
    viz.plotHistory()
