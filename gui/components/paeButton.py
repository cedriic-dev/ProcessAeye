from PyQt5.QtWidgets import QPushButton


class PaeButton(QPushButton):
    def __init__(self, parent=None, x=50, y=50, height=50):
        super().__init__(parent)
        self.x, self.y, self.height = x, y, height
        self.setMinimumSize(200, height)
        self.setGeometry(x, y, 200, height)
        self.setStyleSheet("""
            QPushButton {
                background-color: #111827;
                border: 1px solid transparent;
                border-radius: 10px;
                color: #FFFFFF;
                font-size: 15px;
                font-weight: 600;
                padding: .75rem 1.2rem;
                text-align: center;
                font-family: "Geist Mono Variable Medium";
            }
            QPushButton:hover {
                background-color: #374151;
            }
        """)

    def setText(self, text):
        super().setText(text)
        width = self.fontMetrics().boundingRect(text).width()
        self.setGeometry(self.x, self.y, width, self.height)
