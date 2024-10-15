import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import platform


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, display_width, display_height):
        super().__init__()
        self.display_width = display_width
        self.display_height = display_height

    def run(self):
        if platform.system() == "Linux":
            cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        elif platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while not self.isInterruptionRequested():
            ret, frame = cap.read()
            if ret:
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # Determine dimensions for 1:1 aspect ratio
                height, width, _ = frame.shape
                dimension = min(height, width)

                # Crop the frame to achieve 1:1 aspect ratio
                top = (height - dimension) // 2
                left = (width - dimension) // 2
                cropped_frame = frame[top:top + dimension, left:left + dimension]

                # Resize to the desired dimensions
                resized_frame = cv2.resize(cropped_frame, (self.display_width, self.display_height))

                # Convert BGR to RGB format for QImage
                rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Emit the signal
                self.change_pixmap_signal.emit(convert_to_Qt_format)

    def stop(self):
        self.requestInterruption()  # Request the thread to stop
