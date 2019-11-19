import sys
from scipy.io import loadmat
from pyqtgraph.Qt import QtGui
from app import App

__usage__ = 'Usage: python3 main.py <filename>.mat'

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

    world = data['map']
    goal = data['goal']
    walls = data['walls']
    obs = data['obs']

    app = QtGui.QApplication(['Occupancy Grid Mapping'])
    thisapp = App(walls, obs, goal)
    thisapp.show()
    sys.exit(app.exec_())