#!/usr/bin/python3

import numpy as np
from numpy.random import randn as randn
import control as ctrl
import matplotlib.pyplot as plt
from scipy.io import loadmat

class AUV:
    def __init__(self, discrete_system, process_covariance, sensor_covariance,
            x0=np.zeros((2,1)), ts=0.05):
        self.Ad = np.array(discrete_system.A)
        self.Bd = np.array(discrete_system.B)
        self.C = np.array(discrete_system.C)
        self.x = x0
        self.R_sqrt = np.sqrt(process_covariance)
        self.Q_sqrt = np.sqrt(sensor_covariance)
    
    def propagateDynamics(self, u):
        self.x = self.Ad @ self.x + self.Bd * u + self.R_sqrt @ randn(2,1)
        return self.x

    def getSensorMeasurement(self):
        return self.C @ self.x + self.Q_sqrt @ randn(1)

class KalmanFilter:
    def __init__(self, discrete_system,process_covariance,sensor_covariance):
        self.sigma = np.diag([0.1,1]) # confidence in inital condition
        self.R = process_covariance
        self.Q = sensor_covariance
        self.Ad = np.array(discrete_system.A)
        self.Bd = np.array(discrete_system.B)
        self.C = np.array(discrete_system.C)
        self.ts = discrete_system.dt
        self.mu = np.array([[-2],[2]])

    def predictionStep(self, u):
        self.mu = self.Ad @ self.mu + self.Bd * u
        self.sigma = self.Ad @ self.sigma @ self.Ad.T + self.R
        return self.mu, self.sigma

    def correctionStep(self, z):
        K = self.sigma @ self.C.T * 1/(self.C @ self.sigma @ self.C.T + self.Q)
        self.mu += K * (z - self.C @ self.mu)
        self.sigma = (np.eye(2) - K @ self.C) @ self.sigma
        return self.mu, self.sigma, K

# parameters
mat = loadmat('./hw1_soln_data.mat')
m = 100
b = 20
ts = 0.05

# actual system
A = np.array([[0,1],[0,-b/m]])
B = np.array([[0],[1/m]])
C = np.array([[1,0]])
D = np.array([[0]])
sys = ctrl.ss(A,B,C,D)
sys_d = ctrl.c2d(sys, ts)
R = np.diag([0.0001,0.01])  # process covariance
Q = np.diag([0.001])        # sensor measurement covariance

# dynamics
#states = []
#auv = AUV(sys_d,R,Q,x0=states,ts=ts)

# kalman filter
kf = KalmanFilter(sys_d,R,Q)
estimates = []
pos_2sig = []
vel_2sig = []

# set up plotting variables
pos_error =  []
vel_error =  []
gain = []
time = mat['t'].reshape(1001)
u = mat['u'].reshape(1001)
x_truth = mat['xtr'].reshape(1001)
v_truth = mat['vtr'].reshape(1001)
z = mat['z'].reshape(1001)

# run simulation
for i,t in enumerate(time):

    # propagate actual system
#    x = auv.propagateDynamics(u[i])
#    states.append(x_truth[i])

    # sensor measurement
#    z = auv.getSensorMeasurement()

    # Kalman Filter 
    mu_bar, covariance_bar = kf.predictionStep(u[i])
    mu, covariance, K = kf.correctionStep(z[i])
    if (covariance_bar < covariance).all():
        print('BAD NEWS BEARS') # covariance shrinks with correction step
    estimates.append(mu)
    pos_2sig.append(2*np.sqrt(covariance.item(0)))
    vel_2sig.append(2*np.sqrt(covariance.item(-1)))

    # store plotting variables
    pos_error.append(x_truth[i] - mu.item(0))
    vel_error.append(v_truth[i] - mu.item(1))
    gain.append(K)

estimates = np.array(estimates).reshape(len(estimates),2).T

plt.figure(1)
plt.plot(time, estimates[0,:].T, 'k', label='estimated position')
plt.plot(time, estimates[1,:].T, 'r', label='estimated velocity')
plt.plot(time, x_truth, 'c', label='true position')
plt.plot(time, v_truth, 'b', label='true velocity')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.title('States vs Estimates')
plt.legend()

plt.figure(2)
plt.plot(time, pos_error, label='position error')
plt.plot(time, pos_2sig, 'r', label='pos 2sig')
plt.plot(time, [-j for j in pos_2sig], 'r')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Position Error')
plt.legend()

plt.figure(3)
plt.plot(time, vel_error, label='velocity error')
plt.plot(time, vel_2sig, 'r', label='vel 2sig')
plt.plot(time, [-j for j in vel_2sig], 'r')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Velocity Error')
plt.legend()

gain = np.array(gain).reshape((len(gain),2))
plt.figure(4)
plt.plot(time, gain[:,1], label='velocity gain')
plt.plot(time, gain[:,0], label='position gain')
plt.xlabel('Time (s)')
plt.ylabel('Gain')
plt.title('Kalman Gain over Time')
plt.legend()

plt.show()
