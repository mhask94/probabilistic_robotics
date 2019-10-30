#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
from visualizer import Visualizer
from scipy.io import loadmat

from IPython.core.debugger import set_trace

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

class MotionModel():
    def __init__(self, alphas, ts=0.1, noise=True):
        self.a1, self.a2, self.a3, self.a4 = alphas[:4]
        if len(alphas) == 6:
            self.a5, self.a6 = alphas[4:6]
        else:
            self.a5, self.a6 = 0, 0
        self.dt = ts
        self.noise = noise

    def __call__(self, u, x_m1):
        # add noise if needed
        u_noisy = np.zeros((len(u), len(x_m1[0])))
        vsig = np.sqrt(self.a1*u[0]**2 + self.a2*u[1]**2) * self.noise
        wsig = np.sqrt(self.a3*u[0]**2 + self.a4*u[1]**2) * self.noise
        gamsig = np.sqrt(self.a5*u[0]**2 + self.a6*u[1]**2) * self.noise
        u_noisy[0] = u[0] + vsig*randn(len(x_m1[0]))
        u_noisy[1] = u[1] + wsig*randn(len(x_m1[1]))

        n0, = np.where(u_noisy[1] != 0)    # non-zero indices of omega
        vhat = u_noisy[0] 
        what = u_noisy[1] 
        temp = vhat[n0] / what[n0] 
        w_dt = what * self.dt
        gamma_dt = gamsig*randn(len(w_dt)) * self.dt
        theta = x_m1[2][n0]

        x = np.zeros(x_m1.shape)
        x[0][n0] = x_m1[0][n0] + temp*(np.sin(theta+w_dt[n0])-np.sin(theta))
        x[1][n0] = x_m1[1][n0] + temp*(np.cos(theta)-np.cos(theta+w_dt[n0]))
        x[2] = wrap(x_m1[2] + w_dt + gamma_dt)

        if len(n0) != len(u_noisy[1]): 
            y0, = np.where(u_noisy[1] == 0) # zero indices of omega
            theta = x_m1[2][y0]
            x[0][y0] = x_m1[0][y0] + vhat[y0]*self.dt*np.cos(theta)
            x[1][y0] = x_m1[1][y0] + vhat[y0]*self.dt*np.sin(theta)
        return x

class MeasurementModel():
    def __init__(self):
        pass

    def __call__(self, states, mx, my):
        x_diff, y_diff = mx - states[0], my - states[1]
        r = np.sqrt(x_diff**2 + y_diff**2)
        phi = np.arctan2(y_diff, x_diff) - states[2]
        return np.block([[r], [wrap(phi)]])


class TurtleBot:
    def __init__(self, motion_model, sensor_covariance, x0=np.zeros((3,1)), 
            landmarks=np.empty(0)):
        self.Q_sqrt = np.sqrt(sensor_covariance)
        self.g = motion_model
        self.h = MeasurementModel()
        self.x = wrap(x0, dim=2)
        self.landmarks = landmarks
    
    def propagateDynamics(self, u):
        # propagate through motion_model
        self.x = self.g(u, self.x)
        return self.x

    def getSensorMeasurement(self):
        if not self.landmarks.size > 1:
            return -1 # error if no landmarks provided
        z = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            # add noise to measurement model
            z[:,i] = self.h(self.x, mx, my).flatten() + self.Q_sqrt @ randn(2)
        z[1] = wrap(z[1]) ###
        return z 

class ParticleFilter:
    def __init__(self, motion_model, sensor_covariance, num_particles=1000,
            landmarks=np.empty(0)):
        self.Q = sensor_covariance.diagonal().reshape((len(Q),1))
        self.g = motion_model
        self.h = MeasurementModel()
        self.M = num_particles
        self.landmarks = landmarks
        self.chi = np.empty((4,num_particles))
        self.chi[0] = np.random.uniform(-10, 10, self.M)
        self.chi[1] = np.random.uniform(-10, 10, self.M)
        self.chi[2] = np.random.uniform(-np.pi, np.pi, self.M)
        self.chi[-1] = 1 / num_particles
        self._update_belief()

    def _update_belief(self):
        self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
        self.sigma = np.cov(self.chi[:3])

    def _gauss_prob(self, diff, var):
        return np.exp(-diff**2/2/var) / np.sqrt(2*np.pi*var)

    def _low_var_resample(self):
        M_inv = 1/self.M
        r = np.random.uniform(low=0, high=M_inv)
        c = np.cumsum(self.chi[-1])
        U = np.arange(self.M)*M_inv + r
        diff = c - U[:,None]
        i = np.argmax(diff > 0, axis=1)

        n = 3 # num states

        P = np.cov(self.chi[:n])
        self.chi = self.chi[:,i]

        uniq = np.unique(i).size
        if uniq*M_inv < 0.1:
            Q = P / ((self.M*uniq)**(1/n))
            noise = Q @ randn(*self.chi[:n].shape)
            self.chi[:n] = wrap(self.chi[:n] + noise, dim=2)
        self.chi[-1] = M_inv

    def predictionStep(self, u):
        self.chi[:3] = self.g(u, self.chi[:3])
        self._update_belief()

        return self.mu 

    def correctionStep(self, z):
        z_hat = np.zeros((2,len(self.landmarks))) 
        self.chi[-1] = 1
        for i, (mx,my) in enumerate(self.landmarks):
            Zi = self.h(self.chi[:3], mx, my)
            diff = wrap(Zi - z[:,i:i+1], dim=1)
            z_prob = np.prod(self._gauss_prob(diff, 2*self.Q), axis=0)
            self.chi[-1] *= z_prob
            z_prob /= np.sum(z_prob)
            z_hat[:,i] = np.sum(z_prob * Zi, axis=1)
        self.chi[-1] /= np.sum(self.chi[-1])

        self._low_var_resample()
        self._update_belief()

        return self.mu, self.sigma, z_hat

if __name__ == "__main__":
    ## parameters
    landmarks=np.array([[6,4],[-7,8],[6,-4]])
#    landmarks=np.array([[6,4]])
    alpha = np.array([0.1, 0.01, 0.01, 0.1]) #, 0.01, 0.01])
    Q = np.diag([0.1, 0.05])**2
    M = 1000

    args = sys.argv[1:]
    if len(args) == 0:
        load=False
        ts, tf = 0.1, 20
        time = np.linspace(0, tf, tf/ts+1)
        x0 = np.array([[-5.],[-3.],[np.pi/2]])
    elif len(args) == 1:
        load = True
        mat_file = args[0]
        mat = loadmat(mat_file)
        time = mat['t'][0]
        ts, tf = time[1], time[-1]
        x,y,theta = mat['x'][0], mat['y'][0], mat['th'][0]
        x0 = np.array([[x[0],y[0],theta[0]]]).T
        vn_c, wn_c = mat['v'][0], mat['om'][0]
    else:              
        print('[ERROR] Invalid number of arguments.')

    # inputs
    v_c = 1 + 0.5*np.cos(2*np.pi*0.2*time)
    w_c = -0.2 + 2*np.cos(2*np.pi*0.6*time)

    ## system
    bot_motion_mod = MotionModel(alpha[:4], ts, noise=not load)
    turtlebot = TurtleBot(bot_motion_mod, Q, x0, landmarks)
    
    ## extended kalman filter
    pf_motion_mod = MotionModel(alpha, ts, noise=True)
    pf = ParticleFilter(pf_motion_mod, Q, M, landmarks)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, particles=pf.chi[:2], xhat0=pf.mu,
            sigma0=pf.sigma, landmarks=landmarks, live=True)
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0:
            continue
        # input commands
        u = np.array([v_c[i], w_c[i]]).reshape(2,1)
        if load:
            un = np.array([vn_c[i], wn_c[i]]).reshape(2,1)
        else:
            un = u
    
        # propagate actual system
        x1 = turtlebot.propagateDynamics(un)

        # sensor measurement
        z = turtlebot.getSensorMeasurement()
    
        # Kalman Filter 
        xhat_bar = pf.predictionStep(u)
        xhat, covariance, zhat = pf.correctionStep(z)
    
        # store plotting variables
        viz.update(t, x1, pf.chi[:2], xhat, covariance, zhat)
    
    viz.plotHistory()
