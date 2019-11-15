#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
from visualizer import Visualizer
from scipy.io import loadmat
from turtlebot import TurtleBot
from fast_slam import FastSLAM
from utils import wrap

__usage__ = '\nUsage:\tpython3 main.py <filename>.mat'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == "__main__":
    ## parameters
#    landmarks=np.array([[6,4],[-7,8],[6,-4]])
    lm_per_quad = 3
    ll, hh = 1.5, 9.5
    lmx1 = np.random.uniform(low=ll, high=hh, size=lm_per_quad)
    lmx2 = np.random.uniform(low=-hh, high=-ll, size=lm_per_quad)
    lmx3 = np.random.uniform(low=-hh, high=-ll, size=lm_per_quad)
    lmx4 = np.random.uniform(low=ll, high=hh, size=lm_per_quad)
    lmy1 = np.random.uniform(low=ll, high=hh, size=lm_per_quad)
    lmy2 = np.random.uniform(low=ll, high=hh, size=lm_per_quad)
    lmy3 = np.random.uniform(low=-hh, high=-ll, size=lm_per_quad)
    lmy4 = np.random.uniform(low=-hh, high=-ll, size=lm_per_quad)
    lm1 = np.block([[lmx1],[lmy1]]).T
    lm2 = np.block([[lmx2],[lmy2]]).T
    lm3 = np.block([[lmx3],[lmy3]]).T
    lm4 = np.block([[lmx4],[lmy4]]).T

    landmarks = np.block([[lm1],[lm2],[lm3],[lm4]])
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    Q = np.diag([0.1, 0.05])**2
    sigma = np.diag([1,1,0.1]) # confidence in inital condition
    xhat0 = np.array([[0.],[0.],[0.]]) # changing this causes error initially
    num_particles = 100
    fov = 45.
    avg_type = 'mean'
#    avg_type = 'best'

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
    v_c = 2.0 + 0.1*np.cos(2*np.pi*0.2*time)
    w_c = -0.60 + 0.1*np.cos(2*np.pi*0.6*time)

    ## system
    turtlebot = TurtleBot(alpha, Q, x0=x0, ts=ts, landmarks=landmarks, fov=fov)
    
    ## extended kalman filter
    fast_slam = FastSLAM(alpha, Q, num_particles, len(landmarks), ts, avg_type)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, particles=fast_slam.chi, xhat0=xhat0, 
            sigma0=sigma, landmarks=landmarks, live='True')
    
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

        # Filter 
        xhat_bar = fast_slam.predictionStep(u_c)
        xhat, sig, chi, mu_lm, sig_lm, zhat = fast_slam.correctionStep(z)
    
        # store plotting variables
        viz.update(t, x1, chi, xhat, sig, mu_lm, sig_lm)
    
    viz.plotHistory()
