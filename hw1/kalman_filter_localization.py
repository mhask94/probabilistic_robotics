#!/usr/bin/python3

import numpy as np
from numpy.random import randn as randn
import control as ctrl
import matplotlib.pyplot as plt

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
        self.sigma = np.diag([1,1]) # confidence in inital condition
        self.R = process_covariance
        self.Q = sensor_covariance
        self.Ad = discrete_system.A
        self.Bd = discrete_system.B
        self.C = discrete_system.C
        self.ts = discrete_system.dt
        self.xhat = np.zeros((2,1))

    def predictionStep(self, u):
        self.xhat = self.Ad @ self.xhat + self.Bd * u
        self.sigma = self.Ad @ self.sigma @ self.Ad.T + self.R
        return self.xhat, self.sigma

    def correctionStep(self, z):
        K = self.sigma @ self.C.T * 1/(self.C @ self.sigma @ self.C.T + self.Q)
        self.xhat += K * (z - self.C @ self.xhat)
        self.sigma = (np.eye(2) - K @ self.C) @ self.sigma
        return self.xhat, self.sigma, K

# parameters
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
states = np.array([[0],[0]])
auv = AUV(sys_d,R,Q,x0=states,ts=ts)

# kalman filter
kf = KalmanFilter(sys_d,R,Q)
estimates = np.array([[0],[0]]) # changing this causes error initially
pos_2sig = []
vel_2sig = []

# set up plotting variables
error = states - estimates
gain = []
time = []
t = 0
time.append(t)

tf = 50
mag_force = 50
# run simulation
while t <= tf:
    F = 0
    if t < 5:
        F = mag_force
    elif t >= 25 and t < 30:
        F = -mag_force

    # propagate actual system
    x = auv.propagateDynamics(F)
    states = np.hstack([states,x])

    # sensor measurement
    z = auv.getSensorMeasurement()

    # Kalman Filter 
    xhat_bar, covariance_bar = kf.predictionStep(F)
    xhat, covariance, K = kf.correctionStep(z)
    if (covariance_bar < covariance).all():
        print('BAD NEWS BEARS') # covariance shrinks with correction step
    estimates = np.hstack([estimates, xhat])
    pos_2sig.append(2*np.sqrt(covariance.item(0)))
    vel_2sig.append(2*np.sqrt(covariance.item(-1)))

    # store plotting variables
    error = np.hstack([error,x - xhat])
    gain.append(K)
    
    # increment time
    t += ts
    time.append(t)

plt.figure(1)
plt.plot(time, estimates[0,:].T, 'k', label='estimated position')
plt.plot(time, estimates[1,:].T, 'r', label='estimated velocity')
plt.plot(time, states[0,:].T, 'c', label='true position')
plt.plot(time, states[1,:].T, 'b', label='true velocity')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.title('States vs Estimates')
plt.legend()

plt.figure(2)
plt.plot(time, error[0,:].T, label='position error')
plt.plot(time[1:], pos_2sig, 'r', label='pos 2sig')
plt.plot(time[1:], [-j for j in pos_2sig], 'r')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Position Error')
plt.legend()

plt.figure(3)
plt.plot(time, error[1,:].T, label='velocity error')
plt.plot(time[1:], vel_2sig, 'r', label='vel 2sig')
plt.plot(time[1:], [-j for j in vel_2sig], 'r')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Velocity Error')
plt.legend()

gain = np.array(gain).reshape((len(gain),2))
plt.figure(4)
plt.plot(time[1:], gain[:,0], label='position gain')
plt.plot(time[1:], gain[:,1], label='velocity gain')
plt.xlabel('Time (s)')
plt.ylabel('Gain')
plt.title('Kalman Gain over Time')
plt.legend()

plt.show()
