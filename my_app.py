from cnn import CNN

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from picamera import PiCamera


# class Thread(QThread):
#     def __init__(self, cnn):
#         super(Thread, self).__init__()
#         self.__cnn = cnn
#
#     def run(self) -> None:
#         self.__cnn.train()


class MyApp(QWidget):
    def __init__(self):
        super(MyApp, self).__init__()

        capture_train_label = QLabel("Enter the class name:")
        self.__capture_train_edit = QLineEdit()
        capture_train_button = QPushButton("Capture training image")
        train_button = QPushButton("Train images")
        capture_predict_button = QPushButton("Capture predicting image")
        predict_button = QPushButton("Predict images")
        self.__browser = QTextBrowser()
        clear_button = QPushButton('Clear')

        grid = QGridLayout()
        self.setLayout(grid)
        grid.addWidget(capture_train_label, 0, 0)
        grid.addWidget(self.__capture_train_edit, 1, 0)
        grid.addWidget(capture_train_button, 2, 0)
        grid.addWidget(train_button, 3, 0)
        grid.addWidget(capture_predict_button, 4, 0)
        grid.addWidget(predict_button, 5, 0)
        grid.addWidget(self.__browser, 6, 0)
        grid.addWidget(clear_button, 7, 0)

        self.setWindowTitle('Handwriting Similarity Analysis')
        self.setWindowIcon(QIcon('handwriting_icon.png'))
        self.resize(700, 700)
        self.center()
        self.show()

        self.__camera = PiCamera()
        self.__camera.resolution = (640, 480)
        self.__camera.start_preview(fullscreen=False)

        self.__cnn = CNN(self.__browser)
        # self.__thread = Thread(self.__cnn)

        capture_train_button.clicked.connect(self.capture_train)
        # train_button.clicked.connect(self.__thread.start)
        train_button.clicked.connect(self.__cnn.train)
        capture_predict_button.clicked.connect(self.capture_predict)
        predict_button.clicked.connect(self.__cnn.test)
        clear_button.clicked.connect(self.__browser.clear)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def capture_train(self):
        text = self.__capture_train_edit.text().lower().replace(' ', '_')
        if text == '':
            QMessageBox.information(self, "Class Name", "The class name field is empty.\nPlease enter the class name")
            self.__capture_train_edit.setFocus()
            return
        image_dir = f"train/{text}/{text}_{QDateTime.currentMSecsSinceEpoch()}.png"
        self.__camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")

    def capture_predict(self):
        image_dir = f"predict/{QDateTime.currentMSecsSinceEpoch()}.png"
        self.__camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")
