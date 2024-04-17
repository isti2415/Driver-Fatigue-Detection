import cv2
import numpy as np
import time
from keras.models import load_model
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

def play_alarm_sound():
    pygame.mixer.music.load('alarm.wav')
    pygame.mixer.music.play()

def face_detection(face_cascade, frame, scale_factor):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def preprocess_image(face, img):
    face_rect = cv2.resize(face, (84, 84))
    face_rect = face_rect.astype('float32')
    face_rect /= 255
    return face_rect

class MainApp(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_pretrained_models()
        self.initUI()
        self.last_awake_time = time.time()

    def initUI(self):
        self.setWindowTitle('Driver Drowsiness Detection')
        self.setGeometry(1000, 1000, 1000, 1000)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setFixedSize(1000, 1000)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.show()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            faces = face_detection(self.face_cascade, frame, 1.3)
            if len(faces) == 0:
                current_time = time.time()
                if current_time - self.last_awake_time > 3:
                    play_alarm_sound()
                    cv2.putText(frame, "Wake UP!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Alert!!!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.last_awake_time = time.time()
                
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = frame[y:y+h, x:x+w]
                face_rect = preprocess_image(face, frame)
                face_pred = self.drowsiness_model.predict(np.array([face_rect]))[0]
                if face_pred > 0.5:
                    label = 'Awake'
                    color = (0, 255, 0)
                else:
                    label = 'Drowsy'
                    color = (0, 0, 255)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            image = self.convert_cv_qt(frame).toImage()
            self.label.setPixmap(QPixmap.fromImage(image))
            self.label.setAlignment(Qt.AlignCenter)

    def convert_cv_qt(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg.rgbSwapped())

    def load_pretrained_models(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        drowsiness_model = load_model('model.keras')
        self.face_cascade = face_cascade
        self.drowsiness_model = drowsiness_model

if __name__ == '__main__':
    app = QApplication([])
    main_app = MainApp()
    app.exec_()