import numpy as np
from numpy.random import randn

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle


class MotionModel():
    def __init__(self, motion_cov, ts=0.1, noise=True):
        self.M_sqrt = np.sqrt(motion_cov) # in input space
        self.dt = ts
        self.noise = noise
        self.G = np.eye(3)          # dg / dx
        self.V = np.zeros((3,2))    # dg / du
        self.V[2,1] = self.dt

    # call the class object like a function
    def __call__(self, u, x_m1):
        # add noise if needed
        u_noisy = u + self.M_sqrt @ randn(*x_m1[:2].shape) * self.noise

        v = u_noisy[0]
        w_dt = u_noisy[1] * self.dt
        theta = x_m1[2]
        cos_dt = np.cos(theta) * self.dt
        sin_dt = np.sin(theta) * self.dt

        x = np.zeros(x_m1.shape)
        x[0] = x_m1[0] + v*cos_dt
        x[1] = x_m1[1] + v*sin_dt
        x[2] = wrap(theta + w_dt)

        if len(theta) == 1:
            self.G[0,2], self.G[1,2] = -v*sin_dt, v*cos_dt
            self.V[0,0], self.V[1,0] = cos_dt, sin_dt

        return x

    def jacobians(self):
        return self.G, self.V


class MeasurementModel():
    def __init__(self):
        self.H = np.zeros((2,3))    # dh / dx
        self.H[1,2] = -1

    # call the class object like a function
    def __call__(self, states, mx, my):
        x_diff, y_diff = mx - states[0], my - states[1]
        r = np.sqrt(x_diff**2 + y_diff**2)
        phi = np.arctan2(y_diff, x_diff) - states[2]

        if len(x_diff) == 1:
            self.H[0,0], self.H[0,1] = -x_diff/r, -y_diff/r
            self.H[1,0], self.H[1,1] = y_diff/r**2, -x_diff/r**2

        return np.block([[r], [wrap(phi)]])

    def jacobian(self):
        return self.H
