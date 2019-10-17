#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def wrap(angle):
    angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle

class Visualizer:
    def __init__(self, limits, map0, x0, z0, live=True):
        plt.rcParams["figure.figsize"] = (9,7)
        self.fig, self.ax = plt.subplots()
        self.ax.axis(limits)
        self.ax.set_title('Occupancy Grid Map Simulation')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        x,y,theta = x0.flatten()
        self.R = 1.5
        self.circ = Circle((x,y), radius=self.R, color='y', ec='k')
        self.ax.add_patch(self.circ)
        xdata = [x, x + self.R*np.cos(theta)]
        ydata = [y, y + self.R*np.sin(theta)]
        self.line, = self.ax.plot(xdata, ydata, 'k')

        self.live = live
#        if self.live:
#            self.est_lms, = self.ax.plot(20,20, 'rx', label='est landmark')
#            self.particle_dots, = self.ax.plot(particles[0],particles[1], 'r.',
#                    markersize=2, label='particles')

        self.ax.legend()
        self._display()

    def update(self, pose, z):
        x,y,theta = pose.flatten()
        self.circ.set_center((x,y))
        self.line.set_xdata([x, x + self.R*np.cos(theta)])
        self.line.set_ydata([y, y + self.R*np.sin(theta)])

#        data = np.random.uniform(0,1,(100,100))
#        self.ax.imshow(data , 'Greys')

#        if self.live:
#            self.particle_dots.set_xdata(particles[0])
#            self.particle_dots.set_ydata(particles[1])
#
#            est_lms = np.zeros(zhat.shape)
#            for i, (r,phi) in enumerate(zhat.T):
#                xi = est_pose.item(0) + r*np.cos(phi+theta)
#                yi = est_pose.item(1) + r*np.sin(phi+theta)
#                est_lms[:,i] = np.array([xi,yi])
#            self.est_lms.set_xdata(est_lms[0,:])
#            self.est_lms.set_ydata(est_lms[1,:])
#
        self._display()

    def plotHistory(self):
#        if not self.live:
#            self.true_dots, = self.ax.plot(self.x_hist,self.y_hist, 'b.',
#                    markersize=3, label='truth')
#            self.est_dots, = self.ax.plot(self.xhat_hist,self.yhat_hist, 'r.',
#                    markersize=3, label='estimates')

        plt.rcParams["figure.figsize"] = (8,8)

        self._display()
        input('Press ENTER to close...')

    def _display(self):
        plt.pause(0.000001)
