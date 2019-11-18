import time
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from mdp_planner import MDPPlanner
import numpy as np

class App(QtGui.QMainWindow):
    def __init__(self, walls, obs, goal, parent=None):
        super(App, self).__init__(parent)
        self.map = walls + obs + goal
        self.mdp = MDPPlanner(walls, obs, goal)
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
#            data = 1/(1 + np.exp(self.map.log_prob_map))
            self.img.setImage(data)

#            if self.idx >= len(self.X[0]):
#                self.running = False

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
