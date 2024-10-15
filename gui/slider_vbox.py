import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QSlider, QHBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage, qRgba, QColor, QFont
from PyQt5.QtCore import Qt, QSize, QMutex, QWaitCondition, pyqtSignal

from deep_learning.deepfillv2.app import DeepFillV2
from deep_learning.generative_inpainting.app import GenerativeInpainting
from deep_learning.misf.app import Misf
from gui.components.paeComboBox import PaeComboBox
from gui.components.paeLabel import PaeLabel
from gui.components.paeSlider import PaeSlider
from gui.drawing_widget import DrawingWidget
from gui.inpainting_thread import InpaintingThread
from image_processing.ns.app import NSInpainter
from image_processing.skimage.app import Skimage
from image_processing.telea.app import TeleaInpainter
from utils.IInpaintingAlgorithmen import IInpaintingAlgorithmen


class SliderVbox(QWidget):
    resolution_changed = pyqtSignal(tuple)
    inpainting_algorithm_changed = pyqtSignal(object)

    def __init__(self, drawing_widget, parent=None):
        super(SliderVbox, self).__init__(parent)
        self.resolution_combo_box = PaeComboBox()
        combo_box_font = QFont()
        combo_box_font.setFamily("Geist Mono Variable Medium")
        combo_box_font.setPointSize(16)
        self.resolution_options = {"128x128": (128, 128), "256x256": (256, 256), "512x512": (512, 512)}
        self.resolution_combo_box.addItems(self.resolution_options.keys())
        self.resolution_combo_box.currentIndexChanged.connect(self.emit_resolution_change)
        self.resolution_combo_box.setFont(combo_box_font)

        self.inpaint_algorithm_combo_box = PaeComboBox()
        self.inpaint_algorithms = {"OpenCV TELEA": TeleaInpainter(), "OpenCV NS": NSInpainter(), "Skimage": Skimage(),
                                   "DeepFillV2": DeepFillV2(), "MISF": Misf(),
                                   "Generative Inpainting": GenerativeInpainting()}
        self.inpaint_algorithm_combo_box.addItems(self.inpaint_algorithms.keys())
        self.inpaint_algorithm_combo_box.currentIndexChanged.connect(self.emit_inpainting_algorithm_change)
        self.inpaint_algorithm_combo_box.setFont(combo_box_font)

        self.resolution_combo_box.setFixedSize(200, 50)
        self.inpaint_algorithm_combo_box.setFixedSize(200, 50)

        self.slider_vbox = QVBoxLayout()
        self.brush_size_slider = PaeSlider(self)
        self.brush_size_label = PaeLabel(self)
        self.drawing_widget = drawing_widget
        self.setup_ui()

    def setup_ui(self):
        self.init_slider_vbox()
        self.setLayout(self.slider_vbox)

    def init_slider(self):
        self.brush_size_slider.setRange(1, 25)
        self.brush_size_slider.setMinimumHeight(300)
        self.brush_size_slider.setMaximumHeight(500)
        self.brush_size_slider.setMaximumWidth(50)
        self.brush_size_slider.setValue(int(self.drawing_widget.brush_size))
        self.brush_size_slider.valueChanged.connect(self.change_brush_size)

    def init_slider_vbox(self):
        self.brush_size_label.setText(str(self.drawing_widget.brush_size))
        self.init_slider()
        self.slider_vbox.addWidget(self.inpaint_algorithm_combo_box)
        self.slider_vbox.addWidget(self.resolution_combo_box)  # Add this line
        self.slider_vbox.addWidget(self.brush_size_slider)
        self.slider_vbox.addWidget(self.brush_size_label)
        self.slider_vbox.addStretch(1)

    def change_brush_size(self, value):
        self.drawing_widget.brush_size = value
        self.brush_size_label.setText(str(value))

    def emit_resolution_change(self):
        resolution = self.resolution_options[self.resolution_combo_box.currentText()]
        self.resolution_changed.emit(resolution)

    def emit_inpainting_algorithm_change(self):
        algorithm = self.inpaint_algorithms[self.inpaint_algorithm_combo_box.currentText()]
        self.inpainting_algorithm_changed.emit(algorithm)

    def init_signals(self):
        self.emit_resolution_change()
        self.emit_inpainting_algorithm_change()

    def enable_comobo_boxes(self):
        self.resolution_combo_box.setEnabled(True)
        self.inpaint_algorithm_combo_box.setEnabled(True)

    def disable_combo_boxes(self):
        self.resolution_combo_box.setEnabled(False)
        self.inpaint_algorithm_combo_box.setEnabled(False)