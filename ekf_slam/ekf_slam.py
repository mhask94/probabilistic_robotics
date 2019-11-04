# EKF slam algorithm from Probablistic Robotics Table 10.2

import numpy as np
from utils import wrap, MotionModel, MeasurementModel

from IPython.core.debugger import set_trace

class EKFSlam:
    def __init__(self, alphas, sensor_cov, num_states, num_landmarks, ts=0.1):
        self.g = MotionModel(alphas, noise=False)
        self.h = MeasurementModel()
        self.Q = sensor_cov
        self.dt = ts
        self.n = num_states
        self.N = num_landmarks 
        self.mu_x = np.zeros((num_states, 1))
        self.mu_m = np.zeros((2*num_landmarks, 1))
#        self.mu = np.zeros((self.n+2*self.N, 1))
        self.sig_xx = np.zeros((num_states, num_states))
        self.sig_xm = np.zeros((num_states, 2*num_landmarks))
        self.sig_mm = np.eye(2*num_landmarks) * 100.
#        self.sigma = np.eye(self.n+2*self.N) * 100.

    def predictionStep(self, u):
        self.mu_x = self.g(u, self.mu_x, self.dt)
        Gt, R = self.g.jacobians()
        self.sig_xx = Gt @ self.sig_xx @ Gt.T + R
        self.sig_xm = Gt @ self.sig_xm
        return self.mu_x

    def correctionStep(self, z):
        zhat = np.zeros((2, self.N)) 
        for i in range(self.N):
            if np.isnan(z[0,i]):
                zhat[:,i] = np.nan
                continue
            idx = np.array([2*i, 2*i+1])
            mx, my = self.mu_m[idx]
            if mx == my == 0:
                r,phi = z[:,i]
                mx = r*np.cos(phi+self.mu_x[2]) 
                my = r*np.sin(phi+self.mu_x[2])
                self.mu_m[idx] = mx, my
            zhat[:,i:i+1] = self.h(self.mu_x, mx, my) 
            H1 = self.h.jacobian()
            H2 = -H1[:,:2]

            Hi = np.zeros((2,self.n + 2*self.N))
            Hi[:,:self.n] = H1
            Hi[:,self.n + idx] = H2

            sig_Hi_T = np.block([
                [  self.sig_xx @ H1.T + self.sig_xm[:,idx] @ H2.T],
                [(H1 @ self.sig_xm).T + self.sig_mm[:,idx] @ H2.T]])

            inv = np.linalg.inv(H1 @ sig_Hi_T[:self.n] + 
                    H2 @ sig_Hi_T[self.n+idx] + self.Q)
            Ki = sig_Hi_T @ inv
            innovation = wrap(z[:,i:i+1] - zhat[:,i:i+1], dim=1)

            mu = Ki @ innovation
            self.mu_x += mu[:3]
            self.mu_m += mu[3:]
            self.mu_x = wrap(self.mu_x, dim=2)

            sigma = np.block([[self.sig_xx, self.sig_xm],
                              [self.sig_xm.T, self.sig_mm]])
            sigma = (np.eye(self.n+2*self.N) - Ki @ Hi) @ sigma
            self.sig_xx = sigma[:self.n,:self.n]
            self.sig_xm = sigma[:self.n,self.n:]
            self.sig_mm = sigma[self.n:,self.n:]

        return self.mu_x, self.sig_xx, self.mu_m, self.sig_mm

