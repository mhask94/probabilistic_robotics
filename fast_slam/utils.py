import numpy as np
from numpy.random import randn


def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle


class MotionModel():
    def __init__(self, alphas, noise=True):
        self.a1, self.a2, self.a3, self.a4 = alphas[:4]
        if len(alphas) == 6:
            self.a5, self.a6 = alphas[4:6]
        else:
            self.a5, self.a6 = 0, 0
        self.noise = noise
        self.G = np.eye(3)
        self.R = np.zeros((2,2))

    def __call__(self, u, x_m1, dt):
        # add noise if needed
        u_noisy = np.zeros((len(u), len(x_m1[0])))
        v_var = self.a1*u[0]**2 + self.a2*u[1]**2
        w_var = self.a3*u[0]**2 + self.a4*u[1]**2
        vsig = np.sqrt(v_var) * self.noise
        wsig = np.sqrt(w_var) * self.noise
        gamsig = np.sqrt(self.a5*u[0]**2 + self.a6*u[1]**2) * self.noise
        u_noisy[0] = u[0] + vsig*randn(len(x_m1[0]))
        u_noisy[1] = u[1] + wsig*randn(len(x_m1[1]))

        n0, = np.where(u_noisy[1] != 0)    # non-zero indices of omega
        vhat = u_noisy[0] 
        what = u_noisy[1] 
        temp = vhat[n0] / what[n0] 
        w_dt = what * dt
        gamma_dt = gamsig*randn(len(w_dt)) * dt
        theta = x_m1[2][n0]

        cos_term = np.cos(theta)-np.cos(theta+w_dt[n0])
        sin_term = np.sin(theta+w_dt[n0])-np.sin(theta)

        x = np.zeros(x_m1.shape)
        x[0][n0] = x_m1[0][n0] + temp*sin_term
        x[1][n0] = x_m1[1][n0] + temp*cos_term
        x[2] = wrap(x_m1[2] + w_dt + gamma_dt)

        if len(n0) != len(u_noisy[1]): 
            y0, = np.where(u_noisy[1] == 0) # zero indices of omega
            theta = x_m1[2][y0]
            x[0][y0] = x_m1[0][y0] + vhat[y0]*dt*np.cos(theta)
            x[1][y0] = x_m1[1][y0] + vhat[y0]*dt*np.sin(theta)
        return x

    def jacobians(self):
        return self.G, self.R


class MeasurementModel():
    def __init__(self, num_particles=100, calc_jacobians=False):
        self.N = num_particles
        self.calc_jacobians = calc_jacobians
        if calc_jacobians:
            self.H = np.zeros((self.N,2,2))

    def __call__(self, states, mx, my):
        x_diff, y_diff = mx - states[0], my - states[1]
        r = np.sqrt(x_diff**2 + y_diff**2)
        phi = np.arctan2(y_diff, x_diff) - states[2]

        if self.calc_jacobians:
            self.H[:,0,0], self.H[:,0,1] = x_diff/r, y_diff/r
            self.H[:,1,0], self.H[:,1,1] = -y_diff/r**2, x_diff/r**2

        return np.block([[r], [wrap(phi)]])

    def jacobian(self):
        if self.calc_jacobians:
            return self.H
        else:
            return None
