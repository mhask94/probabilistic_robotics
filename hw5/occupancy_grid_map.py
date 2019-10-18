# This is an implementation of Occupancy Grid Mapping as Presented
# in Chapter 9 of "Probabilistic Robotics" By Sebastian Thrun et al.
# In particular, this is an implementation of Table 9.1 and 9.2

from scipy.linalg import norm
import numpy as np

def wrap(angle, dim=None):
    if dim:
        angle[dim] -= 2*np.pi * np.floor((angle[dim] + np.pi) / (2*np.pi))
    else:
        angle -= 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
    return angle


class OccupancyGridMap():
    '''
    intertial frame: origin @ bottom left of map, x -> right, y -> up
    vehicle frame: origin @ location of vehicle, same rotation as inertial
    body frame: origin @ vehicle, rotated from inertial by theta
    '''
    def __init__(self, xsize, ysize, grid_size):
        self.xsize = xsize 
        self.ysize = ysize
        self.log_prob_map = np.zeros((self.xsize, self.ysize))

        self.alpha = 1.0            # assumed thickness of obstacles
        self.beta = 5.0*np.pi/180.0 # assumed width of the laser beam
        self.z_max = 150.0          # max range of the laser

        # Pre-allocate the x and y of all grid positions into a 3D tensor
        self.cell_coords = np.array(
                [np.tile(np.arange(0, self.xsize*grid_size, 
                     grid_size)[:,None], (1, self.ysize)),
                 np.tile(np.arange(0, self.ysize*grid_size, 
                     grid_size)[:,None].T, (self.xsize, 1))],dtype=np.float64)

        # log-probabilities to add or remove from the map 
        # log-prob = log[p(m) / (1 + p(m))]
        p_occ = 0.65
        p_free = 1 - p_occ
        self.l_occ = np.log(p_occ/p_free)
        self.l_free = np.log(p_free/p_occ)
        self.l0 = np.log(1/(1-0)) # equals 0 (don't update cells out of view)

    def update_map(self, pose, z):
        x,y,theta = pose
        
        dx = self.cell_coords.copy() # inertial frame cell coordinates
        dx[0] -= x # x coordinates of cells in vehicle frame
        dx[1] -= y # y coordinates of cells in vehicle frame

        grid_b = wrap(np.arctan2(dx[1], dx[0]) - theta) # body frame bearings
        grid_r = norm(dx, axis=0) # body frame range to each cell

        # for each laser beam
        for z_i in z:
            if np.sum(np.isnan(z_i)) > 0:
                continue # ignore nans
            r = z_i[0] # measured range 
            b = z_i[1] # measured bearing

            '''
            free_mask: all cells in view of laser beams that are unoccupied
            occ_mask:  all cells in view of laser beams that are occupied
            '''
            free_mask = (np.abs(grid_b - b) <= self.beta/2.0) & \
                (grid_r < min(self.z_max, r - self.alpha/2.0))
            occ_mask = (np.abs(grid_b - b) <= self.beta/2.0) & \
                (np.abs(grid_r - r) <= self.alpha/2.0)

            # apply measurement update to cells in view of laser beams
            self.log_prob_map[occ_mask]  += self.l_occ - self.l0
            self.log_prob_map[free_mask] += self.l_free - self.l0

