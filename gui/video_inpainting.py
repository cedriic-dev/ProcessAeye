import time

import cv2
import pyvirtualcam
from PyQt5.QtCore import QMutex, QWaitCondition, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel

from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT
from gui.base_inpainting import BaseInpainting
from gui.components.paeButton import PaeButton
from gui.components.paeLabel import PaeLabel
from gui.inpainting_thread import InpaintingThread
from gui.slider_vbox import SliderVbox
from gui.video_thread import VideoThread

display_width = DISPLAY_WIDTH
display_height = DISPLAY_HEIGHT


class VideoInpainting(BaseInpainting):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProcessAEye")

        self.main_title = QLabel(self)
        logo_pixmap = QPixmap('gui/components/assets/pae_logo.png')
        self.main_title.setPixmap(logo_pixmap)
        self.main_title.setAlignment(Qt.AlignCenter)

        label_vbox = QVBoxLayout()

        self.fps_samples = []
        self.average_fps = 0
        self.fps_sample_size = 100

        self.image_mutex = QMutex()
        self.image_condition = QWaitCondition()

        self.init_clear_button()
        self.init_inpaint_button()

        self.start_time = None
        self.timer = QTimer(self)
        self.time_label = PaeLabel(" ")
        self.time_label.setStyleSheet("font-size: 24pt;")
        self.fps_label = PaeLabel(" ")
        self.fps_label.setStyleSheet("font-size: 24pt;")
        self.timer.timeout.connect(self.update_elapsed_time)
        self.logo_label = QLabel(self)

        self.last_frame_time = time.time()

        # Layout Boxes
        self.slider_vbox = SliderVbox(self.drawing_widget, self)
        self.original_vbox = QVBoxLayout()  # Vertical box layout
        self.inpaint_vbox = QVBoxLayout()  # Vertical box layout

        self.init_original_vbox()
        self.init_inpainted_vbox()

        self.slider_vbox.resolution_changed.connect(self.set_resolution)
        self.slider_vbox.inpainting_algorithm_changed.connect(self.set_inpaint_algorithm)
        self.slider_vbox.init_signals()

        label_vbox.addWidget(self.time_label, alignment=Qt.AlignCenter)
        label_vbox.addWidget(self.fps_label, alignment=Qt.AlignCenter)
        label_vbox.addStretch(1)

        # Main HBox
        image_hbox = QHBoxLayout()
        image_hbox.addWidget(self.slider_vbox)
        image_hbox.addLayout(self.original_vbox)
        image_hbox.addStretch(1)
        image_hbox.addLayout(label_vbox)
        image_hbox.addStretch(1)
        image_hbox.addLayout(self.inpaint_vbox)

        root = QVBoxLayout()
        root.addWidget(self.main_title, alignment=Qt.AlignCenter)
        root.addLayout(image_hbox)

        self.setLayout(root)
        self.runningVirtualCamera = False
        self.virtual_camera = None
        self.virtual_camera_available = False

        self.init_virtual_camera()
        self.init_start_virtual_camera_button()
        self.check_virtual_camera_availability()

        self.inpaint_vbox.addStretch(1)

        label_vbox.addStretch(1)

        # Start the video thread
        self.thread = VideoThread(display_width, display_height)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        if self.virtual_camera_available:
            self.setLoadingScreen_virtualcamera()

    def init_original_vbox(self):
        original_title = PaeLabel("Original")
        original_title.setFont(self.title_font)

        self.original_vbox.addWidget(original_title)
        self.original_vbox.addWidget(self.image_container)
        self.original_vbox.addWidget(self.clear_button)
        self.original_vbox.addWidget(self.inpaint_button)
        self.original_vbox.addStretch(1)

    def update_image(self, qt_image):
        if self.drawing_widget.pixmap.isNull():
            self.drawing_widget.init_pixmap(qt_image.size())
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def init_inpainted_vbox(self):
        inpaint_title = PaeLabel("Repariertes Bild")
        inpaint_title.setFont(self.title_font)
        self.inpaint_vbox.addWidget(inpaint_title)
        self.inpaint_vbox.addWidget(self.inpaint_image_label)


    def start_threads(self):
        # Start the video thread
        if not self.thread.isRunning():
            self.thread.start()

    def stop_threads(self):
        # Stop the video thread
        if self.thread.isRunning():
            self.thread.stop()

        # Stop the inpainting thread
        if self.inpainting_thread and self.inpainting_thread.isRunning():
            self.inpainting_thread.stop()

    def do_inpaint_image(self):
        if self.inpainting_thread is not None and self.inpainting_thread.isRunning():
            self.inpainting_thread.stop()
            self.inpaint_button.setText("Start")
            self.slider_vbox.enable_comobo_boxes()
            self.last_completed_thread = self.inpainting_thread
            self.inpainting_thread = None
        else:
            self.inpainting_thread = InpaintingThread(self.inpaint_algorithm, self.get_image, self.image_mutex,
                                                      self.image_condition)
            self.inpainting_thread.finished_signal.connect(self.on_inpainting_finished)
            self.inpainting_thread.starting_signal.connect(self.reset_timer)  # Connect the signal
            self.inpainting_thread.start()
            self.inpaint_button.setText("Stop")
            self.slider_vbox.disable_combo_boxes()

    def on_inpainting_finished(self, output_np):
        self.image_mutex.lock()
        dim = display_width, display_height
        resized_image = cv2.resize(output_np, dim, interpolation=cv2.INTER_AREA)
        height, width, channel = resized_image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.inpaint_image_label.setPixmap(QPixmap.fromImage(qt_image))

        self.update_virtual_camera(output_np)
        self.timer.stop()
        self.image_mutex.unlock()
        self.image_condition.wakeAll()
        self.update_fps()

    def setLoadingScreen_virtualcamera(self):
        if not self.virtual_camera_available:
            return
        placeholder_image = cv2.imread("images/A3_Copy_32x.jpg")
        resized_image = cv2.resize(placeholder_image, (512, 512), interpolation=cv2.INTER_AREA)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.flip(resized_image, 1)
        self.virtual_camera.send(resized_image)

    def init_virtual_camera(self):
        try:
            self.virtual_camera = pyvirtualcam.Camera(width=512, height=512, fps=30)
            self.virtual_camera_available = True
            print(f'Benutzte Kamera: {self.virtual_camera.device}')
        except Exception as e:
            self.virtual_camera_available = False
            print(f"Fehler beim Kamera initalisieren: {e}")

    def init_start_virtual_camera_button(self):
        self.start_virtual_camera_button = PaeButton("Virtuelle Kamera starten")
        self.start_virtual_camera_button.clicked.connect(self.toggle_virtual_camera)
        self.inpaint_vbox.addWidget(self.start_virtual_camera_button, alignment=Qt.AlignCenter)

        self.no_driver_label = PaeLabel("Kein Treiber für Virtuelle Kamera vorhanden")
        self.no_driver_label.hide()
        self.inpaint_vbox.addWidget(self.no_driver_label, alignment=Qt.AlignCenter)

        self.inpaint_vbox.addStretch(1)

    def check_virtual_camera_availability(self):
        if not self.virtual_camera_available:
            self.start_virtual_camera_button.setDisabled(True)
            self.no_driver_label.show()

    def update_virtual_camera(self, output_np):
        if self.runningVirtualCamera:
            try:
                print(f'Bild wird gesendet an: {self.virtual_camera.device}')
                if output_np is not None:
                    dim = display_width, display_height
                    resized_image = cv2.resize(output_np, dim, interpolation=cv2.INTER_AREA)
                    flipped_image = cv2.flip(resized_image, 1)
                    self.virtual_camera.send(flipped_image)
                else:
                    print("Kein Bild erhalten")
            except Exception as e:
                print(f"Fehler bei Virtuellen Kamera: {e}")

    def toggle_virtual_camera(self):
        if self.virtual_camera_available:
            self.runningVirtualCamera = not self.runningVirtualCamera
            if self.runningVirtualCamera:
                self.start_virtual_camera_button.setText("Virtuelle Kamera stoppen")
            else:
                self.setLoadingScreen_virtualcamera()
                self.start_virtual_camera_button.setText("Virtuelle Kamera starten")

    def update_elapsed_time(self):
        time.time() - self.start_time
        self.time_label.setText(" ")

    def reset_timer(self):
        self.start_time = time.time()
        self.timer.start(10)

    def update_fps(self):
        current_time = time.time()
        var = 1 / (current_time - self.last_frame_time)
        self.fps_label.setText(" ")
        self.last_frame_time = current_time
