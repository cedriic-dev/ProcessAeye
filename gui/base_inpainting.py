import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import Qt, QMutex, QWaitCondition, QTimer
from PIL import Image

from gui.components.paeButton import PaeButton
from gui.drawing_widget import DrawingWidget
from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen

from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT


def qimage_to_opencv(qimage):
    # Get the size of the QImage
    width = qimage.width()
    height = qimage.height()

    # Get the QImage data
    ptr = qimage.bits()
    ptr.setsize(height * width * 4)

    # Convert the QImage to numpy array
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

    # Extract the RGB channels from the RGBA image
    arr = arr[:, :, :3]

    return arr


class BaseInpainting(QWidget):
    def __init__(self):
        super().__init__()
        self.display_height = DISPLAY_HEIGHT
        self.display_width = DISPLAY_WIDTH
        self.inpaint_algorithm: IInpaintingAlgorithmen = None
        self.inpainting_thread = None
        self.inpainting_height = None
        self.inpainting_width = None
        self.last_completed_thread = None
        self.image_container = QWidget(self)
        self.image_container.setFixedSize(self.display_width, self.display_height)
        self.image_label = QLabel(self.image_container)
        self.image_label.setFixedSize(self.display_width, self.display_height)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: white;")
        self.drawing_widget = DrawingWidget(self.image_container)
        self.drawing_widget.setFixedSize(self.display_width - 4, self.display_height - 4)
        self.drawing_widget.move(2, 2)
        self.inpaint_image_label = QLabel(self)
        self.inpaint_image_label.setFixedSize(self.display_width, self.display_height)
        self.inpaint_image_label.setStyleSheet("border: 2px solid black; background-color: white;")
        self.clear_button = PaeButton(self)
        self.inpaint_button = PaeButton(self)
        self.title_font = QFont("Geist Mono Variable Medium", 20)

        self.image_mutex = QMutex()
        self.image_condition = QWaitCondition()

    def clear_mask(self):
        if not self.drawing_widget.pixmap.isNull():
            self.drawing_widget.init_pixmap(self.image_label.size())
        self.update()

    def clear_image(self):
        self.image_label.clear()

    def clear_inpainting(self):
        self.inpaint_image_label.clear()

    def init_clear_button(self):
        self.clear_button.setText("Zeichnung entfernen")
        self.clear_button.clicked.connect(self.clear_mask)

    def init_inpaint_button(self):
        self.inpaint_button.setText("Start")
        self.inpaint_button.clicked.connect(self.do_inpaint_image)

    def set_resolution(self, res):
        self.inpainting_width, self.inpainting_height = res

    def set_inpaint_algorithm(self, inpainting_algorithmen):
        if self.inpaint_algorithm is not None:
            self.inpaint_algorithm.unload_model()
            current_algorithm = self.inpaint_algorithm.__class__.__name__
            print(current_algorithm + " unloaded")
        self.inpaint_algorithm = inpainting_algorithmen

    def init_drawing_widget(self):
        if self.drawing_widget.pixmap.isNull():
            self.drawing_widget.init_pixmap(self.image_label.size())

    def get_image(self):

        start_time = time.time()

        image_path = "current_image.png"
        mask_path = "current_mask.png"
        image = self.image_label.pixmap().toImage()
        mask = self.drawing_widget.pixmap.toImage()

        image_arr = qimage_to_opencv(image)
        dim = self.inpainting_width, self.inpainting_height
        resizedImage = cv2.resize(image_arr, dim, interpolation=cv2.INTER_AREA)

        if self.inpaint_algorithm.is_deep_learning:
            cv2.imwrite(image_path, resizedImage)
            mask.save(mask_path)
        else:
            # Convert from BGR to RGB for both image and mask
            resized_image_arr = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
            mask_arr = qimage_to_opencv(mask)
            mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2RGB)  # Convert mask to RGB

            image_pil = Image.fromarray(resized_image_arr)
            mask_pil = Image.fromarray(mask_arr)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Image saving and Resizing took: " + str(elapsed_time))

            return image_pil, mask_pil

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Image saving and Resizing took: " + str(elapsed_time))

        return image_path, mask_path
