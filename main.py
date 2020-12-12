from my_app import MyApp
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyApp()
    sys.exit(app.exec_())
