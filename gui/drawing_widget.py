from PyQt5.QtWidgets import QWidget, QGraphicsScene
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QLineF


class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super(DrawingWidget, self).__init__(parent)
        self.drawing = False
        self.last_point = QPoint()
        self.pixmap = QPixmap()  # Keep an empty pixmap here initially
        self.brush_size = 5.0
        self.mask_points = []
        self.history = []

    def init_pixmap(self, size):
        self.pixmap = QPixmap(size)
        self.pixmap.fill(Qt.transparent)  # Fill the pixmap with transparent color

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            pts = {}
            pts['prev'] = (int(self.last_point.x()), int(self.last_point.y()))
            pts['curr'] = (int(event.pos().x()), int(event.pos().y()))
            self.mask_points.append(pts)
            self.history.append(0)
            self.drawMask(self.last_point, event.pos())
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def drawMask(self, prev_pt, curr_pt):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.white, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(QLineF(prev_pt, curr_pt))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def reset(self):
        self.mask_points = []
        self.history = []
        self.last_point = None

    def set_drawing_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()
