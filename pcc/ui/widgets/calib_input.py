import pathlib
from PySide2.QtCore import QEvent, QSize, Qt, Signal, Slot
from PySide2.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QWidget
from .image_view import ImageLabel
from ..image_conversion import pixmapFromNumpy
from ...patterns import PATTERNS

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

def shortenPath(element, path_str):
    """Clips the given text such that it fits within the element's rendered width."""
    width_target = element.width() * 0.95  # We want to keep a minor margin
    width_text = element.fontMetrics().boundingRect(path_str).width()
    if width_text < width_target:
        return path_str

    p = pathlib.Path(path_str)
    if len(p.parts) > 2:
        # Start with the "shortest" path, i.e. first & last entry
        idx_head = 0
        idx_tail = len(p.parts) - 1
        head = [p.parts[idx_head]]
        tail = [p.parts[idx_tail]]
        append_head = False  # We'll alternate adding entries to head and tail (starting with tail)

        # Then we subsequently add path entries until adding the next would
        # overflow the available text space
        text = str(pathlib.Path(*head, '...', *tail))
        width_text = element.fontMetrics().boundingRect(text).width()
        while width_text < width_target and idx_head < idx_tail:  # Also abort if both indices point at the same element (or pass each other)
            if append_head:
                idx_head += 1
                head.append(p.parts[idx_head])
            else:
                idx_tail -= 1
                tail.insert(0, p.parts[idx_tail])
            append_head = not append_head
            text = str(pathlib.Path(*head, '...', *tail))
            width_text = element.fontMetrics().boundingRect(text).width()
        if width_text >= width_target:
            if append_head:  # Last entry has been added to tail
                tail = tail[1:]
            else:  # or head
                head = head[:-1]
            if idx_tail <= idx_head:
                text = str(pathlib.Path(*head, *tail))
            else:
                text = str(pathlib.Path(*head, '...', *tail))
            width_text = element.fontMetrics().boundingRect(text).width()
            if width_text < width_target:
                return text
            else:
                # Use the already shortened path for the fallback/brute-force
                # shortening approach below
                path_str = text
        else:
            return text

        # Simply remove leading characters, one-by-one, until the string is
        # short enough (or we're down to 3 remaining path characters)
        path_str = path_str[3:]
        text = '...' + path_str
        width_text = element.fontMetrics().boundingRect(text).width()
        while width_text > width_target and len(path_str) > 3:  # Keep at least 3 characters
            # Skip an additional character
            path_str = path_str[1:]
            text = '...' + path_str
            width_text = element.fontMetrics().boundingRect(text).width()
        return text


class CalibrationInputWidget(QWidget):
    """UI to select input image folder & configure the calibration target."""

    imageFolderSelected = Signal(str)
    patternConfigurationChanged = Signal()
    
    def __init__(self, icon_size=QSize(20, 20), parent=None):
        super().__init__(parent)
        self._folder = None
        self._initLayout(icon_size)
        self.installEventFilter(self)

    def _initLayout(self, icon_size):
        layout = QGridLayout()
        # 1st row: folder selection
        self._btn_folder = QPushButton(' Open Image Folder')
        self._btn_folder.setIcon(QIcon.fromTheme('document-open'))
        self._btn_folder.setIconSize(icon_size)
        self._btn_folder.setToolTip('Open folder')  #TODO should we register shortcut in main widget (Ctrl+O)?
        self._btn_folder.setMinimumHeight(20)
        self._btn_folder.clicked.connect(self._selectImageFolder)
        layout.addWidget(self._btn_folder, 0, 0, 1, 1, Qt.AlignTop)

        self._lbl_folder = QLabel('')
        self._lbl_folder.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self._lbl_folder.setMinimumWidth(50)
        layout.addWidget(self._lbl_folder, 0, 1, 1, 2, Qt.AlignTop)
        
        # 2nd row: calibration pattern selection, configuration & thumbnail
        self._pattern_selection = QComboBox()
        layout.addWidget(self._pattern_selection, 1, 0, Qt.AlignTop)
        for k in PATTERNS:
            self._pattern_selection.addItem(k)
        self._selected_pattern_idx = 0
        self._pattern_selection.activated.connect(self._patternSelectionChanged)

        self._btn_pattern_config = QPushButton('Configure') #TODO replace by settings wheel
        self._btn_pattern_config.clicked.connect(self._configurePattern)
        layout.addWidget(self._btn_pattern_config, 1, 1, Qt.AlignTop)

        self._thumbnail = ImageLabel(center_vertical=False)
        self._thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._thumbnail, 1, 2, 3, 1)# don't set Qt.AlignTop

        # 3rd row: spacer
        vspace = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addItem(vspace, 2, 0, 1, 2, Qt.AlignTop)
        self.setLayout(layout)

    @Slot(int)
    def _patternSelectionChanged(self, current_index):
        # Reset thumbnail if pattern actually changed
        if current_index != self._selected_pattern_idx:
            self._thumbnail.setPixmap(None)
            self.update()
        # else:
        #     #TODO
        #     import numpy as np
        #     tmp = np.zeros((200, 150, 3), dtype=np.uint8)
        #     tmp[:, :, 2] = 255
        #     self._thumbnail.setPixmap(pixmapFromNumpy(tmp))
        self._selected_pattern_idx = current_index

    @Slot()
    def _configurePattern(self):
        print(f'TODO configure {self._pattern_selection.currentIndex()}')
        import numpy as np
        tmp = np.zeros((200, 150, 3), dtype=np.uint8)
        tmp[:, :, 2] = 255
        self._thumbnail.setPixmap(pixmapFromNumpy(tmp))
        self.update()
        self.patternConfigurationChanged.emit()

    @Slot()
    def _selectImageFolder(self):
        # Let the user select a directory
        selection = str(QFileDialog.getExistingDirectory(self, "Select Calibration Image Directory"))
        if len(selection) > 0 and pathlib.Path(selection).exists():
            self._folder = selection
            self.imageFolderSelected.emit(self._folder)
        self._updateFolderLabel()

    def _updateFolderLabel(self):
        # Display the selected directory path on the label
        txt = '' if self._folder is None else self._folder
        self._lbl_folder.setText(shortenPath(self._lbl_folder, txt))
        self._lbl_folder.update()

    def eventFilter(self, source, event):
        if (event.type() == QEvent.Resize):
            # Adjust text display
            self._updateFolderLabel()
        return super().eventFilter(source, event)

    def resetImageFolder(self):
        """Use this to manually/programmatically clear the selected folder."""
        self._folder = None
        self._updateFolderLabel()
