# Extended Information Filter algorithm from Probablistic Robotics Table 3.5

import numpy as np
from utils import wrap, MotionModel, MeasurementModel

class ExtendedInfoFilter:
    def __init__(self, sensor_cov, motion_cov, ts=0.1, mu0=np.zeros((3,1)),
            sigma0=np.eye(3), landmarks=np.empty(0)):
        self.Q = sensor_cov
        self.Q_inv = np.linalg.inv(sensor_cov)
        self.M = motion_cov
        self.g = MotionModel(motion_cov, ts, noise=False)
        self.h = MeasurementModel()
        self.landmarks = landmarks
        self.mu = mu0
        self.sigma = sigma0
        self.info_mat = np.linalg.inv(self.sigma)
        self.info_vec = self.info_mat @ self.mu

    def predictionStep(self, u):
        # line 2 is done after correction step for plotting reasons
        # do line 5 1st to get jacobians
        self.mu = self.g(u, self.mu)                # line 5 
        Gt, Vt = self.g.jacobians()

        # map noise from control space to state space
        R = Vt @ self.M @ Vt.T 

        self.sigma = Gt @ self.sigma @ Gt.T + R     # line 3 (w/o inverse)
        self.info_mat = np.linalg.inv(self.sigma)   # line 3 (take inverse once)
        self.info_vec = self.info_mat @ self.mu     # line 4

        return self.mu 

    def correctionStep(self, z):
        zhat = np.zeros((2, len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            zhat[:,i:i+1] = self.h(self.mu, mx, my)  # do h(mu) 1st to get Hi
            Hi = self.h.jacobian()
            
            HiT_Qinv = Hi.T @ self.Q_inv
            self.info_mat += HiT_Qinv @ Hi           # line 6

            # DON'T WRAP (Hi * mu)! Lots of issues if you do
            # Do wrap (z - zhat) or else theta error will spike a few times
            innovation = wrap(z[:,i:i+1] - zhat[:,i:i+1], dim=1) + Hi @ self.mu 
            self.info_vec += HiT_Qinv @ innovation   # line 7

            # convert back to moment space for plotting
            self.sigma = np.linalg.inv(self.info_mat)
            self.mu = wrap(self.sigma @ self.info_vec, dim=2)

        return self.info_vec, self.mu, self.sigma, zhat

