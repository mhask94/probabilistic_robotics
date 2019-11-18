# This is a fast SLAM algorith that uses a particle filter for each landmark

import numpy as np
from numpy.random import randn
from utils import wrap, MotionModel, MeasurementModel

class FastSLAM:
    def __init__(self, alphas, sensor_cov, num_particles, num_landmarks, \
            ts=0.1, avg_type='mean'):
        self.g = MotionModel(alphas, noise=True)
        self.h = MeasurementModel(num_particles, calc_jacobians=True)
        self.Q = sensor_cov
        self.dt = ts
        self.N = num_particles
        self.NL = num_landmarks 
        self.chi = np.zeros((4,num_particles))
        self.chi[-1] = 1 / num_particles
        self.mu_m = np.zeros((num_landmarks,num_particles,2,1))
        self.sig_m = np.empty((num_landmarks,num_particles,2,2))
        self.type = avg_type
        self._update_belief()

    def _update_belief(self):
        if self.type == 'mean':
            self.mu = wrap(np.mean(self.chi[:3], axis=1, keepdims=True), dim=2)
            self.mu_lm = np.mean(self.mu_m, axis=1)
            self.sig_lm = np.mean(self.sig_m, axis=1)
        elif self.type == 'best':
            idx = np.argmax(self.chi[-1])
            self.mu = self.chi[:3, idx:idx+1]
            self.mu_lm = self.mu_m[:,idx]
            self.sig_lm = self.sig_m[:,idx]
        self.sigma = np.cov(self.chi[:3])

    def _low_var_resample(self):
        N_inv = 1/self.N
        r = np.random.uniform(low=0, high=N_inv)
        c = np.cumsum(self.chi[-1])
        U = np.arange(self.N)*N_inv + r
        diff = c - U[:,None]
        i = np.argmax(diff > 0, axis=1)

        n = 3 # num states

        P = np.cov(self.chi[:n])
        self.chi = self.chi[:,i]
        self.mu_m = self.mu_m[:,i]
        self.sig_m = self.sig_m[:,i]

        uniq = np.unique(i).size
        if uniq*N_inv < 0.1:
            Q = P / ((self.N*uniq)**(1/n))
            noise = Q @ randn(*self.chi[:n].shape)
            self.chi[:n] = wrap(self.chi[:n] + noise, dim=2)
#        self.chi[-1] = N_inv

    def predictionStep(self, u):
        self.chi[:3] = self.g(u, self.chi[:3], self.dt)
        self._update_belief()
        return self.mu

    def correctionStep(self, z):
        zhat = np.zeros((2, self.NL)) 
        self.chi[-1] = 1
        for i in range(self.NL):
            # don't do anything when no measurement is received to landmark i
            if np.isnan(z[0,i]):
                zhat[:,i] = np.nan
                continue
            # if landmark has never been seen before, initialize it
            mx = self.mu_m[i,:,0].flatten()
            my = self.mu_m[i,:,1].flatten()
            if self.mu_m[i].item(0) == 0:
                r,phi = z[:,i]
                mx = self.chi[0] + r*np.cos(phi+self.chi[2])
                my = self.chi[1] + r*np.sin(phi+self.chi[2])
                self.mu_m[i,:,0] = mx[:,None]
                self.mu_m[i,:,1] = my[:,None]
                Zi = self.h(self.chi[:3], mx, my)
                H_inv = np.linalg.inv(self.h.jacobian())
                self.sig_m[i] = H_inv @ self.Q @ H_inv.transpose((0,2,1))
                continue
            Zi = self.h(self.chi[:3], mx, my) 
            H = self.h.jacobian()
            sig_H_T = self.sig_m[i] @ H.transpose((0,2,1))
            Si = H @ sig_H_T + self.Q
            Si_inv = np.linalg.inv(Si)

            Ki = sig_H_T @ Si_inv
            innov = wrap(z[:,i:i+1] - Zi, dim=1)
            self.mu_m[i] += Ki @ innov.T[:,:,None]
            self.sig_m[i] = (np.eye(2) - Ki @ H) @ self.sig_m[i]
            exp = np.exp(-0.5 * innov.T[:,None,:] @ Si_inv @ innov.T[:,:,None])
            z_prob = np.linalg.det(2*np.pi*Si)**(-0.5) * exp.flatten()
            self.chi[-1] *= z_prob
            # normalize so weights sum to 1
            z_prob /= np.sum(z_prob)
            zhat[:,i] = np.sum(z_prob * Zi, axis=1)

        self.chi[-1] /= np.sum(self.chi[-1])

        self._low_var_resample()
        self._update_belief()

        return self.mu, self.sigma, self.chi, self.mu_lm, self.sig_lm, zhat

