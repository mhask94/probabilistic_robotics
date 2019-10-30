import time
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from occupancy_grid_map import OccupancyGridMap
import numpy as np

class TurtleBotItem(pg.GraphicsObject):
    def __init__(self, pose, radius):
        pg.GraphicsObject.__init__(self)
        self.pose = QtCore.QPointF(*pose[:2])
        self.R = radius 
        pt = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])]) * self.R
        self.pt = QtCore.QPointF(*(pose[:2] + pt))
        self.generatePicture()

    def setPose(self, pose):
        self.pose.setX(pose[0])
        self.pose.setY(pose[1])
        pt = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])]) * self.R
        self.pt.setX(pt[0])
        self.pt.setY(pt[1])
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(QtGui.QPen(Qt.black, 0.5, Qt.SolidLine))
        p.setBrush(QtGui.QBrush(Qt.yellow, Qt.SolidPattern))
        p.drawEllipse(self.pose, self.R, self.R)
        p.drawLine(self.pose, self.pt)
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
        self.map = OccupancyGridMap(100//grid_size, 100//grid_size, grid_size)

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
        self.turtlebot = TurtleBotItem(self.X[:,self.idx], 1.5) 
        self.view.addItem(self.img)
        self.view.addItem(self.turtlebot)


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

            self.turtlebot.setPose(self.X[:,self.idx])

            self.idx += 1
            if self.idx >= len(self.X[0]):
                self.running = False

            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
            QtCore.QTimer.singleShot(1, self._update)
            self.counter += 1
        else:
            tx = 'Finished Mapping!'
        self.label.setText(tx)
