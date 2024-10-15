from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt


class PaeSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setMinimumWidth(20)  # Adjust as needed
        self.setStyleSheet("""
            QSlider::groove:vertical {
                border: 1px solid #1d3557; /* Modern blue color for the groove */
                width: 10px;
                background: #1d3557; /* Modern blue color for the groove */
                margin: 0 2px;
                border-radius: 5px;
            }
            QSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1d3557, stop:1 #374151);
                height: 20px;
                margin: 0 -5px;
                border-radius: 10px;
            }
            QSlider::handle:vertical:hover {
                background: #374151;
            }
            QSlider::sub-page:vertical {
                background: #007BFF;
                border-radius: 5px;
            }
        """)
