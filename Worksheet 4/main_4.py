# -*- coding: utf-8 -*-
import sys
import main_4windownew as mw
import flowmodel_4 as fm

import os

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QApplication, QDialog, QWidget, QMainWindow, QFileDialog
#from PyQt5.uic import loadUiType
from qtpy import uic
from qtpy.uic import loadUi

if __name__ == '__main__':
    ## Create application instance
    
    app = QApplication(sys.argv)

    ## Create and show main window

    widget = mw.MainWindow()
    widget.show()

    ## Start main event loop
    
    sys.exit(app.exec_())