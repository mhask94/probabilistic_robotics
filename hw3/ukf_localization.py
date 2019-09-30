#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
import control as ctrl
from visualizer import Visualizer
from scipy.io import loadmat
import pdb

def wrap(angle):
    angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

class TurtleBot:
    def __init__(self,alphas, sensor_covariance, x0=np.zeros((3,1)),
            ts=0.1, landmarks=np.empty(0)):
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.Q_sqrt = np.sqrt(sensor_covariance)
        self.x = x0
        self.x[2,0] = wrap(self.x[2,0])
        self.dt = ts
        self.landmarks = landmarks
    
    def propagateDynamics(self, u, noise=True):
        vsig = np.sqrt(self.a1*u.item(0)**2 + self.a2*u.item(1)**2) * noise
        wsig = np.sqrt(self.a3*u.item(0)**2 + self.a4*u.item(1)**2) * noise
        vhat = u.item(0) + vsig*randn()
        what = u.item(1) + wsig*randn()
        temp = vhat / what 
        w_dt = what * self.dt
        theta = self.x.item(2)

        if u[1] == 0:
            self.x += np.array([[vhat*self.dt*np.sin(theta)],
                                [vhat*self.dt*np.cos(theta)],
                                [0]])
        else:
            self.x += np.array([[temp*(np.sin(theta+w_dt)-np.sin(theta))],
                                [temp*(np.cos(theta)-np.cos(theta+w_dt))],
                                [w_dt]])
        self.x[2,0] = wrap(self.x[2,0])
        return self.x

    def getSensorMeasurement(self):
        if not self.landmarks.size > 1:
            return -1
        z = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            x_diff, y_diff = mx - self.x.item(0), my - self.x.item(1)
            r = np.sqrt(x_diff**2 + y_diff**2)
            phi = np.arctan2(y_diff, x_diff) - self.x.item(2)
            z[:,i] = np.array([r,phi]) + self.Q_sqrt @ randn(2)
        z[1] = wrap(z[1])
        return z 

class UKF:
    def __init__(self, alphas, sensor_covariance, sigma0=np.eye(3), 
            mu0=np.zeros((3,1)), ts=0.1, landmarks=np.empty(0)):
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.Q = sensor_covariance
        self.sigma = sigma0 
        self.mu = mu0
        self.mu[2,0] = wrap(self.mu[2,0])
        self.dt = ts
        self.landmarks = landmarks
        self.mu_a = np.vstack([self.mu,0,0,0,0])
        self.sigma_a = np.eye(7)
        self.sigma_a[:3,:3] = self.sigma
        self.sigma_a[-2:,-2:] = self.Q
        self.chi_a = np.zeros((7,15))

        alpha = 0.35
        kappa = 3.5
        beta = 2
        n = len(self.mu_a)
        lam = alpha**2 * (n + kappa) - n
        wm_0 = lam / (n + lam)
        wc_0 = wm_0 + (1 - alpha**2 + beta)
        wi = np.ones(14) * (1 / (2*n+2*lam))
        self.wm = np.hstack([wm_0,wi])
        self.wc = np.hstack([wc_0,wi])
        self.gamma = np.sqrt(n+lam)

    def predictionStep(self, u):
        # augmented variables
        M = np.diag([self.a1*u.item(0)**2 + self.a2*u.item(1)**2,
                     self.a3*u.item(0)**2 + self.a4*u.item(1)**2])
        self.mu_a[:3] = self.mu
        self.sigma_a[:3,:3] = self.sigma
        self.sigma_a[3:5,3:5] = M
        L = np.linalg.cholesky(self.sigma_a)
        self.chi_a[:,0] = self.mu_a.flatten()
        self.chi_a[:,1:8] = self.mu_a + self.gamma*L
        self.chi_a[:, 8:] = self.mu_a - self.gamma*L
        # propagate dynamics
        u_a = u + self.chi_a[3:5]
        temp = u_a[0] / u_a[1]
        w_dt = u_a[1] * self.dt
        theta = self.chi_a[2]
        cos_term = np.cos(theta) - np.cos(theta + w_dt)
        sin_term = np.sin(theta + w_dt) - np.sin(theta)
        self.chi_a[:3] += np.block([[temp*sin_term], [temp*cos_term], [w_dt]])
        self.chi_a[2] = wrap(self.chi_a[2])
        # update mu
        self.mu = np.sum(self.wm * self.chi_a[:3], 1, keepdims=True)
        # update sigma
        diff = self.chi_a[:3] - self.mu
        self.sigma *= 0
        for i in range(len(self.chi_a[0])):
            self.sigma += self.wc[i] * np.outer(diff[:,i], diff[:,i])

        return self.mu, self.sigma

    def correctionStep(self, z):
        z_hat = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            x_diff, y_diff = mx - self.chi_a[0], my - self.chi_a[1]
            r_hat = np.sqrt(x_diff**2 + y_diff**2)
            phi_hat = wrap(np.arctan2(y_diff, x_diff) - self.chi_a[2])
            Zi = np.block([[r_hat],[phi_hat]]) + self.chi_a[-2:]
            z_hat[:,i] = np.sum(self.wm * Zi, 1)

            z_diff = Zi - z_hat[:,i].reshape(2,1)
            mu_diff = self.chi_a[:3] - self.mu
            mu_diff[2] = wrap(mu_diff[2])
            Sj = np.zeros((2,2))
            sig_xz = np.zeros((3,2))
            for j in range(len(self.chi_a[0])):
                Sj += self.wc[j] * np.outer(z_diff[:,j], z_diff[:,j])
                sig_xz += self.wc[j] * np.outer(mu_diff[:,j] , z_diff[:,j])
            Sj += self.Q

            Ki = sig_xz @ np.linalg.inv(Sj)
            innov = (z[:,i] - z_hat[:,i]).reshape(2,1)
            innov[1] = wrap(innov[1])
            self.mu += Ki @ innov
            self.mu[2] = wrap(self.mu[2])
            self.sigma -= Ki @ Sj @ Ki.T

            if not i == len(self.landmarks):
                self.mu_a[:3] = self.mu
                self.sigma_a[:3,:3] = self.sigma
                L = np.linalg.cholesky(self.sigma_a)
                self.chi_a[:,0] = self.mu_a.flatten()
                self.chi_a[:,1:8] = self.mu_a + self.gamma*L
                self.chi_a[:, 8:] = self.mu_a - self.gamma*L

        return self.mu, self.sigma, Ki, z_hat

if __name__ == "__main__":
    ## parameters
    landmarks=np.array([[6,4],[-7,8],[6,-4]])
#    landmarks=np.array([[6,4]])
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    Q = np.diag([0.1, 0.05])**2
    sigma = np.diag([1,1,0.1]) # confidence in inital condition
    xhat0 = np.array([[0.],[0.],[0.]]) # changing this causes error initially

    args = sys.argv[1:]
    if len(args) == 0:
        load=False
        ts, tf = 0.1, 20
        time = np.linspace(0, tf, tf/ts+1)
        v_c = 1 + 0.5*np.cos(2*np.pi*0.2*time)
        w_c = -0.2 + 2*np.cos(2*np.pi*0.6*time)
        x0 = np.array([[-5.],[-3.],[np.pi/2]])
    elif len(args) == 1:
        load = True
        mat_file = args[0]
        mat = loadmat(mat_file)
        time = mat['t'][0]
        ts, tf = time[1], time[-1]
        x,y,theta = mat['x'][0], mat['y'][0], mat['th'][0]
        x0 = np.array([[x[0],y[0],theta[0]]]).T
        v_c, w_c = mat['v'][0], mat['om'][0]
    else:              
        print('[ERROR] Invalid number of arguments.')

    ## system
    turtlebot = TurtleBot(alpha, Q, x0=x0, ts=ts, landmarks=landmarks)
    
    ## extended kalman filter
    ukf = UKF(alpha, Q, sigma0=sigma, mu0=xhat0, ts=ts, landmarks=landmarks)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, xhat0=xhat0, sigma0=sigma,
                     landmarks=landmarks, live='True')
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0:
            continue
        # input commands
        u = np.array([v_c[i], w_c[i]]).reshape(2,1)
    
        # propagate actual system
        x1 = turtlebot.propagateDynamics(u, noise=not load)

        # sensor measurement
        z = turtlebot.getSensorMeasurement()
    
        # Kalman Filter 
        xhat_bar, covariance_bar = ukf.predictionStep(u)
        xhat, covariance, K, zhat = ukf.correctionStep(z)
        if (covariance_bar < covariance).all():
            print('BAD NEWS BEARS') # covariance shrinks with correction step
    
        # store plotting variables
        viz.update(t, x1, xhat, covariance, K, zhat)
    
    viz.plotHistory()
