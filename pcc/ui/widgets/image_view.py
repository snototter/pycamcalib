from PySide2.QtCore import QPoint, Qt
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QWidget


class ImageLabel(QWidget):
    """Widget to display an image, always resized to the widgets dimensions."""
    def __init__(self, pixmap=None, center_vertical=True, parent=None):
        super(ImageLabel, self).__init__(parent)
        self._center_vertical = center_vertical
        self._pixmap = pixmap

    def pixmap(self):
        return self._pixmap

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None:
            return
        painter = QPainter(self)
        pm_size = self._pixmap.size()
        pm_size.scale(event.rect().size(), Qt.KeepAspectRatio)
        # Draw resized pixmap using nearest neighbor interpolation instead
        # of bilinear/smooth interpolation (omit the Qt.SmoothTransformation
        # parameter).
        scaled = self._pixmap.scaled(
                pm_size, Qt.KeepAspectRatio)
        pos = QPoint(
            (event.rect().width() - scaled.width()) // 2,
            (event.rect().height() - scaled.height()) // 2 if self._center_vertical else 0)
        painter.drawPixmap(pos, scaled)
