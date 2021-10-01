from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QWidget
from .image_view import ImageLabel
import qimage2ndarray

def pixmapFromNumPy(img_np):
    if img_np.ndim < 3 or img_np.shape[2] in [1, 3, 4]:
        qimage = qimage2ndarray.array2qimage(img_np.copy())
    else:
        img_width = max(400, min(img_np.shape[1], 1200))
        img_height = max(200, min(img_np.shape[0], 1200))
        qimage = QImage(img_width, img_height, QImage.Format_RGB888)
        qimage.fill(Qt.white)
        qp = QPainter()
        qp.begin(qimage)
        qp.setRenderHint(QPainter.HighQualityAntialiasing)
        qp.setPen(QPen(QColor(200, 0, 0)))
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setFamily('Helvetica')
        qp.setFont(font)
        qp.drawText(qimage.rect(), Qt.AlignCenter, "Error!\nCannot display a\n{:d}-channel image.".format(img_np.shape[2]))
        qp.end()
    if qimage.isNull():
        raise ValueError('Invalid image received, cannot convert it to QImage')
    return QPixmap.fromImage(qimage)


class CalibrationPatternSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._initLayout()

    def _initLayout(self):
        layout = QHBoxLayout()
        
        self._combobox = QComboBox()
        layout.addWidget(self._combobox)
        #TODO populate with patterns (load automatically via pcc.patterns, set userData)
        self._combobox.addItem('Checkerboard')
        self._combobox.addItem('Eddie')
        self._combobox.activated.connect(self._slot)

        self._btn_config = QPushButton('Configure')
        layout.addWidget(self._btn_config)

        layout.addStretch()

        self._thumbnail = ImageLabel()
        self._thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._thumbnail)
        self.setLayout(layout)

    @Slot(object)
    def _slot(self, obj):
        print('TODO TODO TODO', type(obj), obj)
        import numpy as np
        tmp = np.zeros((200, 150, 3), dtype=np.uint8)
        tmp[:, :, 2] = 255
        self._thumbnail.setPixmap(pixmapFromNumPy(tmp))
        self.update()
