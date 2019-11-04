#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
from visualizer import Visualizer
from scipy.io import loadmat
from turtlebot import TurtleBot
from ekf_slam import EKFSlam
from utils import wrap

from IPython.core.debugger import set_trace

__usage__ = '\nUsage:\tpython3 main.py <filename>.mat'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == "__main__":
    ## parameters
#    landmarks=np.array([[6,4],[-7,8],[6,-4]])
    num_landmarks = 10
    landmarks = np.random.uniform(low=-10, high=10, size=(num_landmarks,2))
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    Q = np.diag([0.1, 0.05])**2
    sigma = np.diag([1,1,0.1]) # confidence in inital condition
    xhat0 = np.array([[0.],[0.],[0.]]) # changing this causes error initially
    fov = 180.

    args = sys.argv[1:]
    if sys.version_info[0] < 3:
        __error__('Requires Python 3' + __usage__)
    if len(args) == 0:
        load=False
        ts, tf = 0.1, 20
        time = np.linspace(0, tf, tf/ts+1)
        x0 = np.array([[0.],[0.],[0.]])
    elif len(args) == 1:
        load = True
        mat_file = args[0]
        mat = loadmat(mat_file)
        time = mat['t'][0]
        ts, tf = time[1], time[-1]
        x,y,theta = mat['x'][0], mat['y'][0], mat['th'][0]
        x0 = np.array([[x[0],y[0],theta[0]]]).T
        vn, wn = mat['v'][0], mat['om'][0]
    else:              
        __error__('Invalid number of arguments.' + __usage__)

    # inputs
    v_c = 1 + 0.5*np.cos(2*np.pi*0.2*time)
    w_c = -0.25 + 1*np.cos(2*np.pi*0.6*time)

    ## system
    turtlebot = TurtleBot(alpha, Q, x0=x0, ts=ts, landmarks=landmarks, fov=fov)
    
    ## extended kalman filter
    ekf = EKFSlam(alpha, Q, len(x0), len(landmarks), ts=ts)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, xhat0=xhat0, sigma0=sigma,
                     landmarks=landmarks, live='True')
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0:
            continue
        # input commands
        u_c = np.array([v_c[i], w_c[i]])
        if load:
            u = np.array([vn[i], wn[i]])
        else:
            u = u_c
    
        # propagate actual system
        x1 = turtlebot.propagateDynamics(u)

        # sensor measurement
        z = turtlebot.getSensorMeasurement()

#        set_trace()

        # Kalman Filter 
        xhat_bar = ekf.predictionStep(u_c)
        xhat, covariance, zhat = ekf.correctionStep(z)
    
        # store plotting variables
        viz.update(t, x1, xhat, covariance, zhat)
    
    viz.plotHistory()
