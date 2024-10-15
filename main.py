import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from gui.app import App
from gui.video_inpainting import VideoInpainting

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    window.setGeometry(rect)
    window.setWindowTitle("ProcessAeye")
    window.setWindowIcon(QIcon("images/pae_logo_512_512_bgwhite.ico"))
    window.show()
    sys.exit(app.exec_())
