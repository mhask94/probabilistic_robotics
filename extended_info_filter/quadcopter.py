import numpy as np
from numpy.random import randn
from utils import wrap, MotionModel, MeasurementModel

class Quadcopter:
    def __init__(self, sensor_cov, motion_cov, ts=0.1, x0=np.zeros((3,1)), 
            landmarks=np.empty(0)):
        self.Q_sqrt = np.sqrt(sensor_cov)
        self.g = MotionModel(motion_cov, ts, noise=False)
        self.h = MeasurementModel()
        self.x = wrap(x0, dim=2)
        self.landmarks = landmarks
    
    def propagateDynamics(self, u):
        # propagate through motion_model
        self.x = self.g(u, self.x)
        return self.x

    def getSensorMeasurement(self):
        if not self.landmarks.size > 1:
            return -1 # error if no landmarks provided
        z = np.zeros((2,len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            # add noise to measurement model
            z[:,i] = self.h(self.x, mx, my).flatten() + self.Q_sqrt @ randn(2)
        z[1] = wrap(z[1]) ###
        return z 

