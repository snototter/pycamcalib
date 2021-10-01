"""This widget allows to select a folder"""
from PySide2 import QtCore
from PySide2.QtWidgets import QLabel, QSizePolicy, QWidget, QFileDialog, QHBoxLayout, QPushButton
from PySide2.QtCore import Signal, Slot, QSize
from PySide2.QtGui import QIcon
import pathlib


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


class ImageSourceSelectorWidget(QWidget):
    folderSelected = Signal(str)

    def __init__(self, icon_size=QSize(20, 20), parent=None):
        super().__init__(parent)
        self._folder = None
        self.initLayout(icon_size)
        self.installEventFilter(self)

    def initLayout(self, icon_size):
        layout = QHBoxLayout(self)
        # Button
        self._button = QPushButton(' Open folder')
        self._button.setIcon(QIcon.fromTheme('document-open'))
        self._button.setIconSize(icon_size)
        self._button.setToolTip('Open folder')  #TODO should we register shortcut in main widget (Ctrl+O)?
        self._button.clicked.connect(self._selectFolder)
        self._button.setMinimumHeight(20)
        self._button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(self._button)
        # Label to display the selected folder location
        self._label = QLabel('')
        self._label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self._label.setMinimumWidth(30)
        layout.addWidget(self._label)
        # print(self._button.sizeHint(), self._label.sizeHint())

    @Slot()
    def _selectFolder(self):
        # Let the user select a directory
        selection = str(QFileDialog.getExistingDirectory(self, "Select Calibration Directory"))
        if len(selection) > 0 and pathlib.Path(selection).exists():
            self._folder = selection
            self.folderSelected.emit(self._folder)
        self._updateLabel()

    def _updateLabel(self):
        # Display the selected directory path on the label
        txt = '' if self._folder is None else self._folder
        self._label.setText(shortenPath(self._label, txt))
        self._label.update()

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.Resize):
            # Adjust text display
            self._updateLabel()
        return super().eventFilter(source, event)

    def reset(self):
        """Use this to manually/programmatically clear the selected folder."""
        self._folder = None
        self._updateLabel()
