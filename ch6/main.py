import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QComboBox, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("사진 특수 효과")
        self.setGeometry(200, 200, 900, 600)
        self.image = None

        open_button = QPushButton("사진 읽기", self)
        open_button.setGeometry(20, 20, 120, 40)
        open_button.clicked.connect(self.open_image)

        self.effect_combo = QComboBox(self)
        self.effect_combo.setGeometry(160, 20, 160, 40)
        self.effect_combo.addItems(["엠보싱", "카툰", "연필 스케치", "유화"])

        effect_button = QPushButton("선택", self)       
        effect_button.setGeometry(340, 20, 120, 40)
        effect_button.clicked.connect(self.apply_effect)

        quit_button = QPushButton("나가기", self)
        quit_button.setGeometry(480, 20, 120, 40)
        quit_button.clicked.connect(self.close)

        self.image_label = QLabel("사진을 읽어 주세요.", self)
        self.image_label.setGeometry(20, 80, 850, 490)
        self.image_label.setAlignment(Qt.AlignCenter)

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "사진 선택", "", "Images (*.png *.jpg *.jpeg)"
        )
        if filename:
            self.image = cv2.imread(filename)
            self.show_image(self.image)

    def apply_effect(self):
        if self.image is None:
            self.image_label.setText("먼저 사진을 읽어 주세요.")
            return

        choice = self.effect_combo.currentText()

        if choice == "엠보싱":
            kernel = cv2.filter2D(
                self.image, -1,
                cv2.UMat(cv2.getGaussianKernel(3, 0) @
                         cv2.getGaussianKernel(3, 0).T).get()
            )
            result = kernel
        elif choice == "카툰":
            result = cv2.stylization(self.image, sigma_s=60, sigma_r=0.45)
        elif choice == "연필 스케치":
            result, _ = cv2.pencilSketch(self.image, sigma_s=60, sigma_r=0.07)
        else:
            result = cv2.xphoto.oilPainting(self.image, 10, 1)

        self.show_image(result)

    def show_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        qimage = QImage(
            rgb_image.data, width, height, channels * width,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


app = QApplication(sys.argv)
window = SpecialEffect()
window.show()
app.exec_()
