"""Image gallery to show input images, preprocessing & detection results."""

from PySide2.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
from PySide2.QtWidgets import QWidget
from random import randint

class PlaceholderWidget(QWidget):
    """Placeholder during development (until we have proper ImageViewer set up)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255), QColor(0, 0, 0), QColor(255, 255, 255)]
        self.bg_color = colors[randint(0, len(colors)-1)]

    def paintEvent(self, event):
        pen_width = 6
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(pen_width, pen_width, 
                            self.size().width() - 2*pen_width,
                            self.size().height() - 2*pen_width,
                            20, 20)

        pen = QPen(QColor(0, 80, 80), pen_width)
        painter.setPen(pen)
        painter.fillPath(path, self.bg_color)
        painter.drawPath(path)
        # painter.drawRoundedRect(0, 0, self.size().width(), self.size().height(), 10)
        painter.end()
        # return super().paintEvent(event)()

# https://www.pythonguis.com/tutorials/qscrollarea/
# https://stackoverflow.com/questions/20041385/python-pyqt-setting-scroll-area