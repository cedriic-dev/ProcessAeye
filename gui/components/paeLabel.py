from PyQt5.QtWidgets import QLabel


class PaeLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLabel {
            color: black;
            font-family: "Geist Mono Variable Medium";
            }
        """)