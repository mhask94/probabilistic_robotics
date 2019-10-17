import sys
import time
import scipy
from scipy.io import loadmat
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg

class Map():
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
                                         np.tile(np.arange(0, self.ysize*self.grid_size, self.grid_size)[:,None].T, (self.xsize, 1))], dtype=np.float64)

        # Log-Probabilities to add or remove from the map 
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

    def update_map(self, pose, z):

        dx = self.grid_position_m.copy() # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0] # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1] # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2] # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot

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

class CircleItem(pg.GraphicsObject):
    def __init__(self, center, radius):
        pg.GraphicsObject.__init__(self)
        self.center = center
        self.radius = radius
        self.generatePicture()

    def setCenter(self, center):
        self.center = center
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(QtGui.QPen(Qt.black, 0.5, Qt.SolidLine))
        p.setBrush(QtGui.QBrush(Qt.yellow, Qt.SolidPattern))
        p.drawEllipse(self.center[0], self.center[1], self.radius * 2, self.radius * 2)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

class App(QtGui.QMainWindow):
    def __init__(self, X, z, parent=None):
        super(App, self).__init__(parent)
        self.X = X
        self.z = z
        self.idx = 0
        self.running = True
        grid_size = 1
        self.map = Map(100//grid_size, 100//grid_size, grid_size)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 100, 100))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.circ = CircleItem(self.X[:2,self.idx], 1.5) 
        self.view.addItem(self.img)
        self.view.addItem(self.circ)


        #### Set Data  #####################

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):
        if self.running:
            self.map.update_map(self.X[:,self.idx], self.z[:,:,self.idx].T)
            data = 1/(1 + np.exp(self.map.log_prob_map))
            self.img.setImage(data)

            self.circ.setCenter(self.X[:2,self.idx])

            self.idx += 1
            if self.idx >= len(self.X[0]):
                self.running = False
                print('Finished Running')

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1

__usage__ = 'Usage: python3 occupancy_grid_mapping.py <filename>.mat'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()


if __name__ == '__main__':
    args = sys.argv[1:]
    if sys.version_info[0] < 3:
        __error__('Requires Python 3.')

    if len(args) == 1:
        filename = args[0]
        if not filename[-4:] == '.mat':
            __error__('Invalid file extention, expected .mat')
        data = loadmat(filename)
    else:
        __error__('Invalid number of arguments, expected 1.\n' + __usage__)

    pose = data['X']
    thk = data['thk']
    z = data['z']

    app = QtGui.QApplication(['Occupancy Grid Mapping'])
    thisapp = App(pose, z)
    thisapp.show()
    sys.exit(app.exec_())
