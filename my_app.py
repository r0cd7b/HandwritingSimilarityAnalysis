from cnn import CNN
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QTextBrowser, QGridLayout, QDesktopWidget, \
    QMessageBox
from PyQt5.QtGui import QIcon
import time
import os


# from picamera import PiCamera


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        capture_train_label = QLabel('Enter the class name:')
        self.__capture_train_edit = QLineEdit(self)

        capture_train_button = QPushButton("Capture training image")
        capture_train_button.clicked.connect(self.capture_train)

        train_button = QPushButton("Fit image")
        train_button.clicked.connect(self.train)

        capture_predict_button = QPushButton("Capture predicting image")
        capture_predict_button.clicked.connect(self.capture_predict)

        predict_button = QPushButton("Predict image")
        predict_button.clicked.connect(self.predict)

        self.__browser = QTextBrowser()
        self.__browser.setAcceptRichText(True)
        self.__browser.setOpenExternalLinks(True)

        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self.__browser.clear)

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
        self.resize(700, 500)
        self.center()
        self.show()

        self.__cnn = CNN(self.__browser)

        # camera = PiCamera()
        # camera.resolution = (640, 480)
        # camera.start_preview(fullscreen=False)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def capture_train(self):
        text = self.__capture_train_edit.text().lower().replace(' ', '_')
        if text == "":
            QMessageBox.information(self, "Class Name", "The class name field is empty.\nPlease enter the class name")
            self.__capture_train_edit.setFocus()
            return
        image_dir = f"train/{text}/{text}_{time.time_ns()}.png"
        # self.camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")

    def train(self):
        if os.path.exists("model.h5"):
            os.remove("model.h5")
        self.__cnn.fit()

    def capture_predict(self):
        image_dir = f"predict/{time.time_ns()}.png"
        # self.camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")

    def predict(self):
        self.__cnn.predict()
