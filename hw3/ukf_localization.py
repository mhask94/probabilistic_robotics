#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
import control as ctrl
from visualizer import Visualizer
from scipy.io import loadmat

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

class MotionModel():
    def __init__(self, ts=0.1):
        self.dt = ts

    def __call__(self, u, x_m1):
        n0, = np.where(u[1] != 0)    # non-zero indices of omega
        vhat = u[0] 
        what = u[1] 
        temp = vhat[n0] / what[n0] 
        w_dt = what * self.dt
        theta = x_m1[2][n0]

        x = np.zeros(x_m1.shape)
        x[0][n0] = x_m1[0][n0] + temp*(np.sin(theta+w_dt[n0])-np.sin(theta))
        x[1][n0] = x_m1[1][n0] + temp*(np.cos(theta)-np.cos(theta+w_dt[n0]))
        x[2] = wrap(x_m1[2] + w_dt)

        if len(n0) != len(u[1]): 
            print('OMEGA CONTAINS ZEROS')
            y0, = np.where(u[1] == 0) # zero indices of omega
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
        return np.block([[r], [phi]])

class TurtleBot:
    def __init__(self, motion_model, alphas, meas_model, sensor_covariance,
            x0=np.zeros((3,1)), landmarks=np.empty(0)):
        self.g = motion_model
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.h = meas_model
        self.Q_sqrt = np.sqrt(sensor_covariance)
        self.x = wrap(x0, dim=2)
        self.landmarks = landmarks
    
    def propagateDynamics(self, u, noise=True):
        # add noise to commanded inputs
        u_noisy = np.zeros(u.shape)
        vsig = np.sqrt(self.a1*u[0]**2 + self.a2*u[1]**2) * noise
        wsig = np.sqrt(self.a3*u[0]**2 + self.a4*u[1]**2) * noise
        u_noisy[0] = u[0] + vsig*randn(len(vsig))
        u_noisy[1] = u[1] + wsig*randn(len(wsig))
        # propagate through motion_model
        self.x = self.g(u_noisy, self.x)
        return self.x

    def getSensorMeasurement(self):
        if not self.landmarks.size > 1:
            return -1 # error if no landmarks provided
        z = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            # add noise to measurement model
            z[:,i] = self.h(self.x, mx, my).flatten() + self.Q_sqrt @ randn(2)
        z[1] = wrap(z[1]) 
        return z 

class UKF:
    def __init__(self, motion_model, alphas, meas_model, sensor_covariance,
            sigma0=np.eye(3), mu0=np.zeros((3,1)), landmarks=np.empty(0)):
        self.g = motion_model
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.h = meas_model
        self.Q = sensor_covariance
        self.sigma = sigma0 
        self.mu = wrap(mu0, dim=2)
        self.landmarks = landmarks
        self.mu_a = np.vstack([self.mu,0,0,0,0])
        self.sigma_a = np.eye(7)
        self.sigma_a[:3,:3] = self.sigma
        self.sigma_a[-2:,-2:] = self.Q
        self.chi_a = np.zeros((7,15))
        # calculate weights
        alpha = 0.4 # 0.35
        kappa = 4.0 # 3.5
        beta = 2
        n = len(self.mu_a)
        lam = alpha**2 * (n + kappa) - n
        wm_0 = lam / (n + lam)
        wc_0 = wm_0 + (1 - alpha**2 + beta)
        wi = np.ones(14) * (1 / (2*n+2*lam))
        self.wm = np.hstack([wm_0,wi])
        self.wc = np.hstack([wc_0,wi])
        self.gamma = np.sqrt(n+lam)

    def _calcSigmaPoints(self, M=[], update_M=False):
        self.mu_a[:3] = self.mu
        self.sigma_a[:3,:3] = self.sigma
        if update_M:
            self.sigma_a[3:5,3:5] = M
        L = np.linalg.cholesky(self.sigma_a)
        self.chi_a[:,0] = self.mu_a.flatten()
        self.chi_a[:,1:8] = self.mu_a + self.gamma*L
        self.chi_a[:, 8:] = self.mu_a - self.gamma*L
        self.chi_a = wrap(self.chi_a, dim=2)
#        if not update_M: # might not be needed, though it fixed wrapping issues
#            self.chi_a[2] = wrap(self.chi_a[2]) 


    def predictionStep(self, u):
        # augmented variables
        M = np.diag([self.a1*u.item(0)**2 + self.a2*u.item(1)**2,
                     self.a3*u.item(0)**2 + self.a4*u.item(1)**2])
        self._calcSigmaPoints(M, update_M=True)
        u_a = u + self.chi_a[3:5]
        # propagate dynamics through motion model
        self.chi_a[:3] = self.g(u_a, self.chi_a[:3])
        # update mu
        self.mu = np.sum(self.wm * self.chi_a[:3], 1, keepdims=True)
        # update sigma
        diff = wrap(self.chi_a[:3] - self.mu, dim=2)
        self.sigma = np.einsum('ij,kj->ik', self.wc*diff, diff)

        return self.mu, self.sigma

    def correctionStep(self, z):
        z_hat = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            Zi = self.h(self.chi_a[:3], mx, my) + self.chi_a[-2:]
            z_hat[:,i] = np.sum(self.wm * Zi, 1)

            z_diff = wrap(Zi - z_hat[:,i].reshape(2,1), dim=1)
            mu_diff = wrap(self.chi_a[:3] - self.mu, dim=2)
            Sj = np.einsum('ij,kj->ik', self.wc*z_diff, z_diff)
            sig_xz = np.einsum('ij,kj->ik', self.wc*mu_diff, z_diff)

            Ki = sig_xz @ np.linalg.inv(Sj)
            innov = wrap((z[:,i] - z_hat[:,i]).reshape(2,1), dim=1)
            self.mu = wrap(self.mu + Ki @ innov, dim=2)
            self.sigma -= Ki @ Sj @ Ki.T

            if not i == len(self.landmarks):
                self._calcSigmaPoints()

        return self.mu, self.sigma, Ki, z_hat

if __name__ == "__main__":
    ## parameters
    landmarks=np.array([[6,4],[-7,8],[6,-4]])
#    landmarks=np.array([[6,4]])
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    Q = np.diag([0.1, 0.05])**2
    sigma = np.diag([1,1,0.1]) # confidence in inital condition
    xhat0 = np.array([[-4.5],[-3.7],[np.pi/2-0.05]]) 

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
#    v_c = 1 + 0.5*np.cos(2*np.pi*0.2*time)
#    w_c = -0.2 + 2*np.cos(2*np.pi*0.6*time)
    v_c = 1 + 0.5*np.sin(2*np.pi*0.3*time)
    w_c = -0.2 + 2*np.cos(2*np.pi*1.0*time)

    # models
    motion_mod = MotionModel(ts=ts)
    meas_mod = MeasurementModel()

    ## system
    turtlebot = TurtleBot(motion_mod, alpha, meas_mod, Q, x0, landmarks)
    
    ## extended kalman filter
    ukf = UKF(motion_mod, alpha, meas_mod, Q, sigma, xhat0, landmarks)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, xhat0=xhat0, sigma0=sigma,
                     landmarks=landmarks, live=True)
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0:
            continue
        # input commands
        uc = np.array([v_c[i], w_c[i]]).reshape(2,1)
        if load:
            u = np.array([vn_c[i], wn_c[i]]).reshape(2,1)
        else:
            u = uc
    
        # propagate actual system
        x1 = turtlebot.propagateDynamics(u, noise=not load)

        # sensor measurement
        z = turtlebot.getSensorMeasurement()
    
        # Kalman Filter 
        xhat_bar, covariance_bar = ukf.predictionStep(uc)
        xhat, covariance, K, zhat = ukf.correctionStep(z)
        if (covariance_bar < covariance).all():
            print('BAD NEWS BEARS') # covariance shrinks with correction step
    
        # store plotting variables
        viz.update(t, x1, xhat, covariance, K, zhat)
    
    viz.plotHistory()
