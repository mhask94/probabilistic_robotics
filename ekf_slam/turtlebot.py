import numpy as np
from numpy.random import randn
from utils import wrap, MotionModel, MeasurementModel

def rad(degree):
    return degree * np.pi/180

class TurtleBot:
    def __init__(self,alphas, sensor_covariance, x0=np.zeros((3,1)),
            ts=0.1, landmarks=np.empty(0), fov=360):
        self.g = MotionModel(alphas, noise=True)
        self.h = MeasurementModel()
        self.Q_sqrt = np.sqrt(sensor_covariance)
        self.x = wrap(x0, dim=2)
        self.dt = ts
        self.landmarks = landmarks
        self.bearing_lim = rad(fov/2)
    
    def propagateDynamics(self, u):
        self.x = self.g(u, self.x, self.dt)
        return self.x

    def getSensorMeasurement(self):
        if not self.landmarks.size > 1:
            return -1
        z = np.zeros((2, len(self.landmarks))) 
        for i, (mx,my) in enumerate(self.landmarks):
            z[:,i:i+1] = self.h(self.x, mx, my) + self.Q_sqrt @ randn(2,1)
            z[1] = wrap(z[1])

        unseen = np.abs(z[1]) > self.bearing_lim
        z[:,unseen] = np.nan

        return z 

