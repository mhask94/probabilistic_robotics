import time
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from mdp_planner import MDPPlanner
import numpy as np
from utils import ArrowItem, LineItem

class App(QtGui.QMainWindow):
    def __init__(self, walls, obs, goal, start, parent=None):
        super(App, self).__init__(parent)
        w, o, g, m = -100, -5000, 1e5, -2
        self.start = start
        self.map = np.zeros((*obs.shape,3))
        self.map[:,:,2] = walls * g
        self.map[:,:,0] = obs * g
        self.map[:,:,1] = goal * g
        self.mask = (walls + obs + goal) == 0
        self.mask[0] = self.mask[-1] = self.mask[:,0] = self.mask[:,-1] = False
        self.mdp = MDPPlanner(walls, obs, goal, w, o, g, m)
        self.idx = 0
        self.running = True
        self.min_delta = obs.size / 2

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
        self.view.setRange(QtCore.QRectF(0,0,*self.mask.T.shape )) #size screen

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
            self.map[self.mask,:] = data[self.mask,None]

            self.img.setImage(self.map.transpose((1,0,2))[:,::-1])

            if self.mdp.count > 350 or delta < self.min_delta:
                self.running = False
                self.final_iter = self.mdp.count
                self.final_del = delta

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
            print('Calculating optimal policy')
            tx = 'Finished Planning.\tIter: {}\tDelta: {}'
            tx = tx.format(self.final_iter, self.final_del)

            # plot policy function (arrows)
            policy = np.flip(self.mdp.pi.T, 1)
            nr, nc = policy.shape
            for i in range(2, nr-2):
                for j in range(2, nc-2):
                    if not np.flip(self.mask.T,1)[i,j]:
                        continue
                    arrow = ArrowItem(np.array([i,j]), policy[i,j])
                    self.view.addItem(arrow)
            
            # plot optimal policy (magenta line)
            si, sj = self.start[1], len(self.mdp.V[0])-self.start[0]-1
            i = 0
            prev = (si, sj)
            prev2 = prev[:]
            while np.flip(self.mask.T, 1)[si,sj]:
                p_ij = policy[si,sj]
                line = LineItem(np.array([si,sj]), p_ij)
                self.view.addItem(line)
                if p_ij == 0:
                    sj += 1
                elif p_ij == 1:
                    si += 1
                elif p_ij == 2:
                    sj -= 1
                elif p_ij == 3:
                    si -= 1
                if (si, sj) == prev2:
                    break
                prev2 = prev[:]
                prev = (si, sj)
                i += 1
                if i > self.mask.size:
                    break

        self.label.setText(tx)
