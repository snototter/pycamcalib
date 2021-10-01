"""Common/basic UI widgets & functionality."""
from PySide2.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QProgressBar, QSizePolicy, QWidget


def displayMessage(icon_type: int, title: str, text: str, informative_text: str, parent=None):
    msg = QMessageBox(parent)
    msg.setIcon(icon_type)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.exec_()


def displayError(text: str, title: str = 'Error', informative_text: str = '', parent=None):
    displayMessage(QMessageBox.Critical, title, text, informative_text, parent)
