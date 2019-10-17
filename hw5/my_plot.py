import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg

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
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

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
        self.circ = CircleItem(np.array([50,50]), 1.5) 
        self.view.addItem(self.img)
        self.view.addItem(self.circ)

#        self.canvas.nextRow()
        #  line plot
#        self.otherplot = self.canvas.addPlot()
#        self.h2 = self.otherplot.plot(pen='y')


        #### Set Data  #####################
        self.pos = np.array([5.,5.])

        self.x = np.linspace(0,50., num=100)
        self.X,self.Y = np.meshgrid(self.x,self.x)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):

#        self.data = np.sin(self.X/3.+self.counter/9.)*np.cos(self.Y/3.+self.counter/9.)
        self.data = np.ones((100,100,3)) * np.random.rand(100,100,1)
        if self.counter % 100 == 0: 
            self.pos += 0.1 
        self.circ.setCenter(self.pos)
#        print('X size: ', self.X.shape)
#        print('Y size: ', self.Y.shape)
#        print('data size: ', self.data.shape)
#        print('max: ', np.max(self.data))
#        print('min: ', np.min(self.data))
#        exit()
        self.ydata = np.sin(self.x/3.+ self.counter/9.)

        self.img.setImage(self.data)
#        self.h2.setData(self.ydata)

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


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
