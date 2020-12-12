from my_app import MyApp
from cnn import CNN
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    application = QApplication(sys.argv)
    cnn = CNN()
    my_app = MyApp(cnn)
    sys.stdout = my_app
    sys.exit(application.exec_())
