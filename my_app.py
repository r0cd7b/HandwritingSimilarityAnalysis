from cnn import CNN
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import time
import os


# from picamera import PiCamera


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.train_label = QLabel('Enter class name:')
        self.train_edit = QLineEdit(self)

        self.capture_train_button = QPushButton("Capture training image")
        self.capture_train_button.clicked.connect(self.capture_train)

        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train)

        self.capture_predict_button = QPushButton("Capture predicting image")
        self.capture_predict_button.clicked.connect(self.capture_predict)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        self.browser = QTextBrowser()
        self.browser.setAcceptRichText(True)
        self.browser.setOpenExternalLinks(True)

        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.browser.clear)

        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.grid.addWidget(self.train_label, 0, 0)
        self.grid.addWidget(self.train_edit, 1, 0)
        self.grid.addWidget(self.capture_train_button, 2, 0)
        self.grid.addWidget(self.train_button, 3, 0)
        self.grid.addWidget(self.capture_predict_button, 4, 0)
        self.grid.addWidget(self.predict_button, 5, 0)
        self.grid.addWidget(self.browser, 6, 0)
        self.grid.addWidget(self.clear_button, 7, 0)

        self.setWindowTitle('Handwriting Similarity Analysis')
        self.setWindowIcon(QIcon('handwriting_icon.png'))
        self.resize(700, 500)
        self.center()
        self.show()

        self.cnn = CNN(self.browser, "train")

        # self.camera = PiCamera()
        # self.camera.resolution = (640, 480)
        # self.camera.start_preview(fullscreen=False, window=(100, 100, 640, 480))

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def capture_train(self):
        text = self.train_edit.text()
        if text == "":
            QMessageBox.question(
                self,
                'The class name is empty',
                'Please write the class name',
                QMessageBox.Yes,
                QMessageBox.Yes
            )
            return
        image_dir = f"train/{text.replace(' ', '_')}/{time.time()}.png"
        # self.camera.capture(image_dir)
        self.browser.append(f"The image was saved in {image_dir}")

    def train(self):
        if os.path.exists("model.h5"):
            os.remove("model.h5")
        self.cnn = CNN(self.browser, "train")

    def capture_predict(self):
        image_dir = f"predict/{time.time()}.png"
        # self.camera.capture(image_dir)
        self.browser.append(f"The image was saved in {image_dir}")

    def predict(self):
        if self.cnn is None:
            return
        self.cnn.predict("predict")
