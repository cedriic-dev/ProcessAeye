import time
import cv2
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt, QWaitCondition, QMutex
from gui.base_inpainting import BaseInpainting
from gui.components.paeButton import PaeButton
from gui.inpainting_thread import InpaintingThread
from gui.slider_vbox import SliderVbox
from gui.video_thread import VideoThread


def is_near_white(color):
    threshold = 240  # this value can be adjusted based on the exact shade of white in your mask
    return color.red() > threshold and color.green() > threshold and color.blue() > threshold


class SingleImageInpainting(BaseInpainting):
    def __init__(self):
        super().__init__()

        self.num_executs = 0

        self.image_mutex = QMutex()
        self.image_condition = QWaitCondition()
        self.timer = QTimer(self)
        self.time_label = QLabel("Inpainting Time: -- seconds", self)
        self.slider_box = SliderVbox(self.drawing_widget, self)
        self.upload_button = PaeButton("Upload Image")
        self.upload_mask = PaeButton("Upload Mask")

        self.start_time = None
        self.inpainting_thread = None
        self.layout = None

        self.setup_ui()  # This sets up the UI for the SingleImageInpainting widget

    def setup_ui(self):
        self.layout = self.create_single_inpaint_page()  # Create the upload page and get its layout
        self.setLayout(self.layout)

    def create_single_inpaint_page(self):
        root = QVBoxLayout()

        main_headline = QFont("Arial", 32, QFont.Bold)
        main_title = QLabel("Single Image Inpainting")
        main_title.setFont(main_headline)

        self.timer.timeout.connect(self.update_elapsed_time)
        self.time_label.setStyleSheet("font-size: 24pt;")

        vBox = QVBoxLayout()
        image_vbox = QVBoxLayout()
        inpaint_vbox = QVBoxLayout()
        main_hbox = QHBoxLayout()

        self.init_clear_button()
        self.init_inpaint_button()

        self.slider_box.resolution_changed.connect(self.set_resolution)
        self.slider_box.inpainting_algorithm_changed.connect(self.set_inpaint_algorithm)
        self.slider_box.init_signals()

        original_title = QLabel("Original")
        original_title.setFont(self.title_font)

        image_vbox.addWidget(original_title, alignment=Qt.AlignCenter)
        image_vbox.addWidget(self.image_container, alignment=Qt.AlignCenter)
        image_vbox.addWidget(self.clear_button, alignment=Qt.AlignCenter)
        image_vbox.addWidget(self.inpaint_button, alignment=Qt.AlignCenter)
        image_vbox.addStretch(1)

        inpaint_title = QLabel("Inpainted")
        inpaint_title.setFont(self.title_font)

        inpaint_vbox.addWidget(inpaint_title, alignment=Qt.AlignCenter)
        inpaint_vbox.addWidget(self.inpaint_image_label, alignment=Qt.AlignCenter)
        inpaint_vbox.addStretch(1)

        main_hbox.addWidget(self.slider_box)
        main_hbox.addLayout(image_vbox)
        main_hbox.addStretch(1)
        main_hbox.addWidget(self.time_label, alignment=Qt.AlignTop)
        main_hbox.addStretch(1)
        main_hbox.addLayout(inpaint_vbox)

        vBox.addWidget(main_title, alignment=Qt.AlignCenter)
        vBox.addLayout(main_hbox)

        self.upload_button.clicked.connect(self.upload_image)
        vBox.addWidget(self.upload_button, alignment=Qt.AlignCenter)  # Center the button

        self.upload_mask.clicked.connect(self.upload_mask_method)
        vBox.addWidget(self.upload_mask, alignment=Qt.AlignCenter)  # Center the button

        root.addLayout(vBox)

        self.init_drawing_widget()

        return root

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.display_width, self.display_height))

    def upload_mask_method(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name).scaled(self.display_width, self.display_height)
            qimage = pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format_ARGB32)
            transparent_color = QColor(0, 0, 0, 0)

            # Process the image to retain only near-white pixels
            for y in range(qimage.height()):
                for x in range(qimage.width()):
                    if not is_near_white(qimage.pixelColor(x, y)):
                        qimage.setPixelColor(x, y, transparent_color)

            # Convert back to QPixmap
            processed_pixmap = QPixmap.fromImage(qimage)

            self.drawing_widget.set_drawing_pixmap(processed_pixmap)

    def do_inpaint_image(self):
        first_time = False
        if self.num_executs == 0:
            first_time = True

        if self.inpainting_thread is None:
            self.inpainting_thread = InpaintingThread(self.inpaint_algorithm, self.get_image, self.image_mutex,
                                                      self.image_condition, False, first_time)
            self.inpainting_thread.finished_signal.connect(self.on_inpainting_finished)
            self.start_time = time.time()
            self.timer.start(10)
            self.inpainting_thread.start()
            self.disable_ui()

    def on_inpainting_finished(self, output_np):
        self.image_mutex.lock()
        dim = self.display_width, self.display_height
        resized_image = cv2.resize(output_np, dim, interpolation=cv2.INTER_AREA)
        height, width, channel = resized_image.shape
        bytes_per_line = 3 * width

        qt_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.inpaint_image_label.setPixmap(QPixmap.fromImage(qt_image))
        self.image_mutex.unlock()
        self.image_condition.wakeAll()
        self.last_completed_thread = self.inpainting_thread
        self.inpainting_thread = None
        self.timer.stop()
        self.enable_ui()  # Enable the UI only after all runs are complete

        # Increment the execution counter and store the elapsed time
        self.num_executs += 1

    def update_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        self.time_label.setText(f"Inpainting Time: {elapsed_time:.3f} seconds")

    def disable_ui(self):
        self.slider_box.disable_combo_boxes()
        self.inpaint_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.upload_mask.setEnabled(False)

    def enable_ui(self):
        self.slider_box.enable_comobo_boxes()
        self.inpaint_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.upload_mask.setEnabled(True)

    def clear_image_and_mask(self):
        self.clear_mask()
        self.clear_image()
        self.clear_inpainting()
