from my_app import MyApp

import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    application = QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(application.exec_())
