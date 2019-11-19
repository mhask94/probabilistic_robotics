import time
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from mdp_planner import MDPPlanner
import numpy as np

from IPython.core.debugger import set_trace

class ArrowItem(pg.GraphicsObject):
    def __init__(self, origin, direction):
        pg.GraphicsObject.__init__(self)
        points = ArrowItem.pts(direction) + origin.reshape(2,1)
        self.p1 = QtCore.QPointF(*points[:,0])
        self.p2 = QtCore.QPointF(*points[:,1])
        self.p3 = QtCore.QPointF(*points[:,2])
        self.p4 = QtCore.QPointF(*points[:,3])
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(QtGui.QPen(Qt.yellow, 0.1, Qt.SolidLine))
#        p.setBrush(QtGui.QBrush(Qt.yellow, Qt.SolidPattern))
        p.drawLine(self.p1, self.p2)
        p.drawLine(self.p3, self.p2)
        p.drawLine(self.p4, self.p2)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

    @staticmethod
    def pts(direction):
        points = np.array([[0, -.4],[0, .4],[-.2, .3],[.2, .3]]).T
        angle = -np.pi/2 * direction
        C_ang, S_ang = np.cos(angle), np.sin(angle)
        R = np.array([[C_ang, -S_ang], [S_ang, C_ang]])
        points = R @ points + np.array([[.5,.5]]).T
        return points

class App(QtGui.QMainWindow):
    def __init__(self, walls, obs, goal, parent=None):
        super(App, self).__init__(parent)
        w, o, g = -100, -5000, 1e5
        self.map = np.zeros((*obs.shape,3))
        self.map[:,:,2] = walls * g
        self.map[:,:,0] = obs * g
        self.map[:,:,1] = goal * g
        self.mask = (walls + obs + goal) == 0
        self.mask[0] = self.mask[-1] = self.mask[:,0] = self.mask[:,-1] = False
#        self.mask[:,0] = self.mask[:,-1] = False
#        walls + obs + goal
        self.mdp = MDPPlanner(walls, obs, goal, w, o, g)
        self.idx = 0
        self.running = True

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
        self.view.addItem(self.img)


        #### Set Data  #####################

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):
        if self.running:
            delta = self.mdp.update()
            data = self.mdp.V
#            data[self.mask] /= np.max(data[self.mask])
#            data[self.mask] = np.exp(data[self.mask])
#            data[self.mask] /= np.sum(data[self.mask])
            self.map[self.mask,:] = data[self.mask,None]

            self.img.setImage(self.map)

#            self.running = False

            if self.mdp.count > 350 or delta < 100:
                self.running = False

            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate: {fps:.3f} FPS\tIter: {itr}\tDelta: {dlt:.3f}'
            tx = tx.format(fps=self.fps, itr=self.mdp.count, dlt=delta)
            QtCore.QTimer.singleShot(1, self._update)
            self.counter += 1
        else:
            tx = 'Finished Planning'
#            arrow = ArrowItem(np.array([75,95]), 1)
#            self.view.addItem(arrow)

            policy = self.mdp.pi
            for i in range(2, len(policy)-2):
                for j in range(2, len(policy[0])-2):
                    if not self.mask[i,j]:
                        continue
                    arrow = ArrowItem(np.array([i,j]), policy[i,j])
                    self.view.addItem(arrow)

        self.label.setText(tx)
