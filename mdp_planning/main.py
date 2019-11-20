import sys
from scipy.io import loadmat
from pyqtgraph.Qt import QtGui
from app import App

__usage__ = 'Usage: python3 main.py <filename>.mat start_x start_y'

def __error__(msg):
    print('[ERROR] ' + msg)
    exit()

if __name__ == '__main__':
    args = sys.argv[1:]
    if sys.version_info[0] < 3:
        __error__('Requires Python 3.')

    if len(args) == 3:
        filename = args[0]
        if not filename[-4:] == '.mat':
            __error__('Invalid file extention, expected .mat')
        data = loadmat(filename)
        start = (int(args[1]), int(args[2]))
    else:
        __error__('Invalid number of arguments.\n' + __usage__)

    world = data['map']
    goal = data['goal']
    walls = data['walls']
    obs = data['obs']

    app = QtGui.QApplication(['MDP Planning'])
    thisapp = App(walls, obs, goal, start)
    thisapp.show()
    sys.exit(app.exec_())
