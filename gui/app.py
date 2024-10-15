import cv2
from PyQt5.QtGui import QPixmap, QFontDatabase
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, \
    QSpacerItem, QSizePolicy, QComboBox, QFileDialog, QStackedWidget, QLabel

from gui.components.paeButton import PaeButton
from gui.drawing_widget import DrawingWidget
from gui.sinlge_image_inpainting import SingleImageInpainting

from gui.video_inpainting import VideoInpainting


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup stacked widget
        self.stacked_widget = QStackedWidget()

        self.setStyleSheet("background-color: #fff")
        self.showMaximized()

        font_db = QFontDatabase()
        font_db.addApplicationFont("gui/components/assets/fonts/GeistMonoVariableVF.ttf")

        # Add pages to stacked widget
        self.video_inpainting_page = VideoInpainting()
        self.stacked_widget.addWidget(self.video_inpainting_page)
        self.single_image_inpainting = SingleImageInpainting()
        self.stacked_widget.addWidget(self.single_image_inpainting)

        # Navigation
        self.nav_widget = QWidget()
        self.nav_layout = QVBoxLayout()

        self.switch_page_btn = PaeButton("Switch to Single Inpainting")
        self.switch_page_btn.clicked.connect(self.on_switch_page_clicked)
        self.nav_layout.addWidget(self.switch_page_btn)

        self.nav_widget.setLayout(self.nav_layout)

        # Set main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.stacked_widget)
        self.main_layout.addWidget(self.nav_widget)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.logo_label = QLabel(self.nav_widget)
        self.logo_pixmap = QPixmap("./components/assets/pae_logo.png")
        self.logo_label.setPixmap(self.logo_pixmap)
        self.nav_layout.insertWidget(0, self.logo_label)

        self.stacked_widget.currentChanged.connect(self.on_page_changed)

    def on_page_changed(self, index):
        # Assuming VideoInpainting is the first page (index 0)
        if index == 0:
            self.video_inpainting_page.start_threads()
            self.single_image_inpainting.clear_image_and_mask()
        else:
            self.video_inpainting_page.stop_threads()

    def on_switch_page_clicked(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index == 0:
            self.stacked_widget.setCurrentIndex(1)
            self.switch_page_btn.setText("Switch to Video Inpainting")
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.switch_page_btn.setText("Switch to Single Image Inpaint")

    def closeEvent(self, event):
        self.video_inpainting_page.stop_threads()
        if self.video_inpainting_page.last_completed_thread is not None:
            self.video_inpainting_page.last_completed_thread.show_graph()
        elif self.single_image_inpainting.last_completed_thread is not None:
            self.single_image_inpainting.last_completed_thread.show_graph()
