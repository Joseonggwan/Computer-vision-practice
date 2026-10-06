import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel


def make_beep(filename, duration):
    sample_rate = 44100
    frequency = 880
    volume = 0.35
    sample_count = int(sample_rate * duration)

    with wave.open(filename, "wb") as sound_file:
        sound_file.setnchannels(1)
        sound_file.setsampwidth(2)
        sound_file.setframerate(sample_rate)

        for i in range(sample_count):
            value = int(
                volume * 32767 *
                math.sin(2 * math.pi * frequency * i / sample_rate)
            )
            sound_file.writeframesraw(struct.pack("<h", value))


class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("삑 소리 내기")
        self.setGeometry(200, 200, 500, 150)

        self.sound_dir = tempfile.TemporaryDirectory()
        self.short_file = os.path.join(self.sound_dir.name, "short.wav")
        self.long_file = os.path.join(self.sound_dir.name, "long.wav")
        self.player = None

        make_beep(self.short_file, 0.5)
        make_beep(self.long_file, 3)

        short_button = QPushButton("짧게 삑", self)
        short_button.setGeometry(10, 10, 120, 35)
        short_button.clicked.connect(self.play_short)

        long_button = QPushButton("길게 삑", self)
        long_button.setGeometry(140, 10, 120, 35)
        long_button.clicked.connect(self.play_long)

        quit_button = QPushButton("나가기", self)
        quit_button.setGeometry(270, 10, 120, 35)
        quit_button.clicked.connect(self.close)

        self.label = QLabel("버튼을 눌러 소리를 재생하세요.", self)
        self.label.setGeometry(10, 65, 450, 35)

    def play_sound(self, filename, message):
        if self.player and self.player.poll() is None:
            self.player.terminate()

        self.label.setText(message)
        self.player = subprocess.Popen(["afplay", filename])

    def play_short(self):
        self.play_sound(self.short_file, "0.5초 동안 삑 소리를 냅니다.")

    def play_long(self):
        self.play_sound(self.long_file, "3초 동안 삑 소리를 냅니다.")

    def closeEvent(self, event):
        if self.player and self.player.poll() is None:
            self.player.terminate()
        self.sound_dir.cleanup()
        event.accept()


app = QApplication(sys.argv)
window = BeepSound()
window.show()
app.exec_()
