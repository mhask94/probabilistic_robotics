#!/usr/bin/python3

import sys
import numpy as np
from numpy.random import randn as randn
import control as ctrl
from visualizer import Visualizer
from scipy.io import loadmat

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
        vsig = np.sqrt(self.a1*u[0]**2 + self.a2*u[1]**2) * noise
        wsig = np.sqrt(self.a3*u[0]**2 + self.a4*u[1]**2) * noise
        vhat = u[0] + vsig*randn()
        what = u[1] + wsig*randn()
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
        z = np.zeros((len(self.landmarks), 2)) 
        for i, (mx,my) in enumerate(self.landmarks):
            x_diff, y_diff = mx - self.x.item(0), my - self.x.item(1)
            r = np.sqrt(x_diff**2 + y_diff**2)
            phi = np.arctan2(y_diff, x_diff) - self.x.item(2)
            z[i] = np.array([r,phi]) + randn(1,2) @ self.Q_sqrt
            z[i,1] = wrap(z[i,1])
        return z 

class EKF:
    def __init__(self, alphas, sensor_covariance, sigma0=np.eye(3), 
            mu0=np.zeros((3,1)), ts=0.1, landmarks=np.empty(0)):
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.Q = sensor_covariance
        self.sigma = sigma0 
        self.mu = mu0
        self.mu[2,0] = wrap(self.mu[2,0])
        self.dt = ts
        self.G = np.eye(3)
        self.landmarks = landmarks

    def predictionStep(self, u):
        temp = u[0] / u[1]
        w_dt = u[1] * self.dt
        theta = self.mu.item(2)
        cos_term = np.cos(theta) - np.cos(theta + w_dt)
        sin_term = np.sin(theta + w_dt) - np.sin(theta)
        # g(u_t, mu_{t-1})
        self.mu += np.array([[temp*sin_term], [temp*cos_term], [w_dt]])
        self.mu[2,0] = wrap(self.mu[2,0])
        # g' or G
        self.G[0,2] = -temp*cos_term
        self.G[1,2] = -temp*sin_term
        # V
        V = np.array([[sin_term/u[1], -temp*sin_term/u[1]],
                      [cos_term/u[1], -temp*cos_term/u[1]],
                      [0, self.dt]])
        M = np.diag([self.a1*u[0]**2 + self.a2*u[1]**2,
                     self.a3*u[0]**2 + self.a4*u[1]**2])
        self.sigma = self.G @ self.sigma @ self.G.T + V @ M @ V.T
        return self.mu, self.sigma

    def correctionStep(self, z):
        z_hat = np.zeros((len(self.landmarks),2)) 
        for i, (mx,my) in enumerate(self.landmarks):
            x_diff, y_diff = mx - self.mu.item(0), my - self.mu.item(1)
            r_hat = np.sqrt(x_diff**2 + y_diff**2)
            phi_hat = wrap(np.arctan2(y_diff, x_diff) - self.mu.item(2))
            z_hat[i] = np.array([r_hat,phi_hat]) 

            Hi = np.array([[-x_diff/r_hat, -y_diff/r_hat, 0],
                           [y_diff/r_hat**2, -x_diff/r_hat**2, -1]])
            Si = Hi @ self.sigma @ Hi.T + self.Q
            Ki = self.sigma @ Hi.T @ np.linalg.inv(Si)
            innov = (z[i] - z_hat[i]).reshape(2,1)
            innov[1,0] = wrap(innov[1,0])
            self.mu += Ki @ innov
            self.mu[2,0] = wrap(self.mu[2,0])
            self.sigma = (np.eye(3) - Ki @ Hi) @ self.sigma

        return self.mu, self.sigma, Ki, z_hat

if __name__ == "__main__":
    ## parameters
    landmarks=np.array([[6,4],[-7,8],[6,-4]])
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    Q = np.diag([0.1, 0.05])**2
    sigma = np.diag([1,1,0.1]) # confidence in inital condition
    xhat0 = np.array([[0.],[0.],[0.]]) # changing this causes error initially

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
    turtlebot = TurtleBot(alpha, Q, x0=x0, ts=ts, landmarks=landmarks)
    
    ## extended kalman filter
    ekf = EKF(alpha, Q, sigma0=sigma, mu0=xhat0, ts=ts, landmarks=landmarks)
    
    # plotting
    lims=[-10,10,-10,10]
    viz = Visualizer(limits=lims, x0=x0, xhat0=xhat0, sigma0=sigma,
                     landmarks=landmarks, live='True')
    
    # run simulation
    for i,t in enumerate(time):
        if i == 0:
            continue
        # input commands
        u = [v_c[i], w_c[i]]
        if load:
            un = [vn_c[i], wn_c[i]]
        else:
            un = u
    
        # propagate actual system
        x1 = turtlebot.propagateDynamics(un, noise=not load)

        # sensor measurement
        z = turtlebot.getSensorMeasurement()
    
        # Kalman Filter 
        xhat_bar, covariance_bar = ekf.predictionStep(u)
        xhat, covariance, K, zhat = ekf.correctionStep(z)
        if (covariance_bar < covariance).all():
            print('BAD NEWS BEARS') # covariance shrinks with correction step
    
        # store plotting variables
        viz.update(t, x1, xhat, covariance, K, zhat)
    
    viz.plotHistory()
