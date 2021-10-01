from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QPushButton, QSizePolicy, QSpacerItem, QWidget
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


# class CalibrationPatternSelector(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)

#         self._initLayout()

#     def _initLayout(self):
#         layout = QGridLayout()
        
#         self._combobox = QComboBox()
#         layout.addWidget(self._combobox, 0, 0, Qt.AlignTop)
#         #TODO populate with patterns (load automatically via pcc.patterns, set userData)
#         self._combobox.addItem('Checkerboard')
#         self._combobox.addItem('Eddie')
#         self._combobox.activated.connect(self._slot)

#         self._btn_config = QPushButton('Configure')
#         layout.addWidget(self._btn_config, 0, 1, Qt.AlignTop)

#         self._thumbnail = ImageLabel()
#         self._thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         layout.addWidget(self._thumbnail, 0, 2, 2, 1)

#         # vspace = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
#         # layout.addWidget(vspace, 1, 0, 1, 2, Qt.AlignTop)
        
#         self.setLayout(layout)

#     @Slot(object)
#     def _slot(self, obj):
#         print('TODO TODO TODO', type(obj), obj)
#         import numpy as np
#         tmp = np.zeros((200, 150, 3), dtype=np.uint8)
#         tmp[:, :, 2] = 255
#         self._thumbnail.setPixmap(pixmapFromNumPy(tmp))
#         self.update()

from .input_source import ImageSourceSelectorWidget
class CalibrationPatternSelector(QWidget): #TODO rename
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initLayout()

    def _initLayout(self):
        layout = QGridLayout()
        
        # 1st row: folder selection & calibration pattern thumbnail
        # self.image_selector = ImageSourceSelectorWidget()
        # self.image_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # layout.addWidget(self.image_selector, 0, 0, 1, 2, Qt.AlignTop)
        self._button = QPushButton(' Open folder')
        self._button.setIcon(QIcon.fromTheme('document-open'))
        layout.addWidget(self._button, 0, 0, 1, 1, Qt.AlignTop)

        self._thumbnail = ImageLabel()
        self._thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._thumbnail, 0, 2, 3, 1)#, Qt.AlignTop)

        # 2nd row: calibration pattern selection & configuration
        self._combobox = QComboBox()
        layout.addWidget(self._combobox, 1, 0, Qt.AlignTop)
        #TODO populate with patterns (load automatically via pcc.patterns, set userData)
        self._combobox.addItem('Checkerboard')
        self._combobox.addItem('Eddie')
        self._combobox.activated.connect(self._slot)

        self._btn_config = QPushButton('Configure')
        layout.addWidget(self._btn_config, 1, 1, Qt.AlignTop)

        vspace = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addItem(vspace, 2, 0, 1, 2, Qt.AlignTop)
        # self.setMinimumHeight(200)
        self.setLayout(layout)

    @Slot(object)
    def _slot(self, obj):
        print('TODO TODO TODO', type(obj), obj)
        import numpy as np
        tmp = np.zeros((200, 150, 3), dtype=np.uint8)
        tmp[:, :, 2] = 255
        self._thumbnail.setPixmap(pixmapFromNumPy(tmp))
        self.update()

