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
    def __init__(self, xsize, ysize, grid_size):
        self.xsize = xsize+2 # Add extra cells for the borders
        self.ysize = ysize+2
        self.grid_size = grid_size # save this off for future use
        self.log_prob_map = np.zeros((self.xsize, self.ysize)) # set all to zero

        self.alpha = 1.0 # The assumed thickness of obstacles
        self.beta = 5.0*np.pi/180.0 # The assumed width of the laser beam
        self.z_max = 150.0 # The max reading from the laser

        # Pre-allocate the x and y positions of all grid positions into a 3D tensor
        # (pre-allocation = faster)
        self.grid_position_m = np.array([np.tile(np.arange(0, self.xsize*self.grid_size, self.grid_size)[:,None], (1, self.ysize)),
                                         np.tile(np.arange(0, self.ysize*self.grid_size, self.grid_size)[:,None].T, (self.xsize, 1))],dtype=np.float64)

        # Log-Probabilities to add or remove from the map 
        p_occ = 0.65
        p_free = 1 - p_occ
        self.l_occ = np.log(p_occ/p_free)
        self.l_free = np.log(p_free/p_occ)

    def update_map(self, pose, z):
        x,y,theta = pose

        dx = self.grid_position_m.copy() # A tensor of coordinates of all cells
        dx[0, :, :] -= x # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= y # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - theta # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
#        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
#        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi
        theta_to_grid = wrap(theta_to_grid)

        dist_to_grid = norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        # For each laser beam
        for z_i in z:
            r = z_i[0] # range measured
            b = z_i[1] # bearing measured

            # Calculate which cells are measured free or occupied, so we know which cells to update
            # Doing it this way is like a billion times faster than looping through each cell (because vectorized numpy is the only way to numpy)
            free_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (dist_to_grid < (r - self.alpha/2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (np.abs(dist_to_grid - r) <= self.alpha/2.0)

            # Adjust the cells appropriately
            self.log_prob_map[occ_mask] += self.l_occ
            self.log_prob_map[free_mask] += self.l_free

