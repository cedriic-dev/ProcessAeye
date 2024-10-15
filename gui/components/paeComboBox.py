from PyQt5.QtWidgets import QComboBox


class PaeComboBox(QComboBox):
    def __init__(self, parent=None, x=50, y=50, height=50):
        super().__init__(parent)
        self.x, self.y, self.height = x, y, height
        self.setGeometry(x, y, 200, height)
        self.setMinimumSize(200, height)
        self.setStyleSheet("""
            QComboBox {
                background-color: #111827;
                border: 1px solid transparent;
                border-radius: 10px;
                color: #FFFFFF;
                font-size: 15px;
                font-weight: 600;
                padding: .75rem 1.2rem;
                align-items: center;
                font-family: "Geist Mono Variable Medium";
            }
            QComboBox:hover {
                background-color: #374151;
            }
            QComboBox::drop-down {
                width: 30px;
                border: none;
                color: white;
            }
            QComboBox::down-arrow {
                image: url("gui/components/assets/arrow.png");
            }
        """)