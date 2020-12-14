from my_app import MyApp

import sys
from PyQt5.QtWidgets import QApplication

# main 함수로 GUI App 생성 후 작업 수행
if __name__ == '__main__':
    application = QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(application.exec_())
