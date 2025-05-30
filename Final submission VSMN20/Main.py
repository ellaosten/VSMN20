# -*- coding: utf-8 -*-
import sys
import MainWindow as mw

import sys

from qtpy.QtWidgets import QApplication
from qtpy.uic import loadUi

if __name__ == '__main__':
    # Create application instance
    
    app = QApplication(sys.argv)

    # Create and show main window

    widget = mw.MainWindow()
    
    widget.show()

    # Start main event loop
    
    sys.exit(app.exec_())