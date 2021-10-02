"""Common/basic UI widgets & functionality."""
from PySide2.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QSizePolicy, QWidget

# QIcon.fromTheme: use system theme icons, see naming specs at
# https://specifications.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html

def displayMessage(icon_type: int, title: str, text: str, informative_text: str, parent=None):
    msg = QMessageBox(parent)
    msg.setIcon(icon_type)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.exec_()


def displayError(text: str, title: str = 'Error', informative_text: str = '', parent=None):
    displayMessage(QMessageBox.Critical, title, text, informative_text, parent)


def ignoreMessageCallback(text: str, timeout: int = 0):
    """Dummy message 'display' if we don't want to pass the status bar callback
    to the main window's child widgets."""
    pass


class HLine(QFrame):
    """A horizontal line (divider)."""
    def __init__(self, parent=None):
        super(HLine, self).__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
