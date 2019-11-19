from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

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

class LineItem(pg.GraphicsObject):
    def __init__(self, origin, direction):
        pg.GraphicsObject.__init__(self)
        points = LineItem.pts(direction) + origin.reshape(2,1)
        self.p1 = QtCore.QPointF(*points[:,0])
        self.p2 = QtCore.QPointF(*points[:,1])
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(QtGui.QPen(Qt.magenta, 0.2, Qt.SolidLine))
        p.drawLine(self.p1, self.p2)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

    @staticmethod
    def pts(direction):
        points = np.array([[0, 0],[0, 1]]).T
        angle = -np.pi/2 * direction
        C_ang, S_ang = np.cos(angle), np.sin(angle)
        R = np.array([[C_ang, -S_ang], [S_ang, C_ang]])
        points = R @ points + np.array([[.5,.5]]).T
        return points

