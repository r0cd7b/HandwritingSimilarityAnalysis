from cnn import CNN

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from picamera import PiCamera


# 학습 과정을 다른 Thread로 수행하기 위한 Class
# class Thread(QThread):
#     def __init__(self, cnn):
#         super(Thread, self).__init__()
#         self.__cnn = cnn
#
#     def run(self) -> None:
#         self.__cnn.train()


# GUI 구현 Class
class MyApp(QWidget):
    def __init__(self):
        super(MyApp, self).__init__()

        # 각종 Flame 정의
        capture_train_label = QLabel("Enter the class name:")
        self.__capture_train_edit = QLineEdit()
        capture_train_button = QPushButton("Capture training image")
        train_button = QPushButton("Train images")
        capture_predict_button = QPushButton("Capture predicting image")
        predict_button = QPushButton("Predict images")
        self.__browser = QTextBrowser()
        clear_button = QPushButton('Clear')

        # Grid Layout으로 Flame 배열
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

        # Main 창 열기
        self.setWindowTitle('Handwriting Similarity Analysis')
        self.setWindowIcon(QIcon('handwriting_icon.png'))
        self.resize(700, 700)
        self.center()
        self.show()

        # Picamera 작동
        self.__camera = PiCamera()
        self.__camera.resolution = (640, 480)
        self.__camera.start_preview(fullscreen=False, window=(100, 100, 640, 480))

        # CNN 객체 생성
        self.__cnn = CNN(self.__browser)
        # self.__thread = Thread(self.__cnn)  # CNN 객체를 Thread 객체에 넣어 생성

        capture_train_button.clicked.connect(self.capture_train)
        train_button.clicked.connect(self.__cnn.train)
        # train_button.clicked.connect(self.__thread.start)
        capture_predict_button.clicked.connect(self.capture_predict)
        predict_button.clicked.connect(self.__cnn.test)
        clear_button.clicked.connect(self.__browser.clear)

    # 초기 Main 창을 중앙에 열기 위한 함수
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    # Train image 촬영을 수행하는 함수
    def capture_train(self):
        text = self.__capture_train_edit.text().lower().replace(' ', '_')
        if text == '':
            QMessageBox.information(self, "Class Name", "The class name field is empty.\nPlease enter the class name")
            self.__capture_train_edit.setFocus()
            return
        image_dir = f"train/{text}/{text}_{QDateTime.currentMSecsSinceEpoch()}.png"
        self.__camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")

    # Predict image 촬영을 수행하는 함수
    def capture_predict(self):
        image_dir = f"predict/{QDateTime.currentMSecsSinceEpoch()}.png"
        self.__camera.capture(image_dir)
        self.__browser.append(f"The image was saved as {image_dir}")
