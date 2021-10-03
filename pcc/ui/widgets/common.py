"""Common/basic UI widgets & functionality."""
from PySide2.QtCore import QLocale, Qt, Signal, Slot
from PySide2.QtGui import QDoubleValidator, QFontDatabase, QValidator
from PySide2.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QSizePolicy, QWidget

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


class HorizontalLine(QFrame):
    """A horizontal line (divider)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class VerticalLine(QFrame):
    """A horizontal line (divider)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)



###### Input Validation
class ValidatedFloatInputWidget(QWidget):
    editingFinished = Signal()
    valueChanged = Signal()

    def __init__(self, label_text: str, initial_value: float = None,
                 min_val: float = None, max_val: float = None, decimals: int = None,
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        # Label to the left:
        lbl = QLabel(label_text)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl.setAlignment(Qt.AlignLeft)
        layout.addWidget(lbl)
        # Input to the right
        self.line_edit = QLineEdit()
        self.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.line_edit.setAlignment(Qt.AlignRight)
        self.line_edit.editingFinished.connect(self.editingFinished)
        layout.addWidget(self.line_edit)
        # Add input validation to allow only floating point numbers. 
        # Additionally, force '.' as decimal point (instead of comma) by
        # overriding the validator's locale:
        locale = QLocale(QLocale.C)
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        self.is_valid = True
        self.validator = QDoubleValidator()
        self.validator.setLocale(locale)
        if min_val is not None:
            self.validator.setBottom(min_val)
        if max_val is not None:
            self.validator.setTop(max_val)
        if decimals is not None:
            self.validator.setDecimals(decimals)
        self.line_edit.setValidator(self.validator)
        self.line_edit.textEdited.connect(self._textEdited)
        if initial_value is not None:
            fmt = '{:' + (f'.{decimals}' if decimals is not None else '') + 'f}'
            text = fmt.format(initial_value)
            self.line_edit.setText(text)
            self._textEdited(text)

    def valid(self):
        return self.is_valid

    def value(self):
        if self.is_valid:
            return float(self.line_edit.text())
        else:
            return None

    @Slot(str)
    def _textEdited(self, text):
        res, modified_text, _ = self.validator.validate(text, 0)
        # print(f'Validated "{text}": {res}', self.validator.bottom(), self.validator.top())

        if res == QValidator.Acceptable:
            self.is_valid = True
            val = float(modified_text)
            self.line_edit.setStyleSheet("border: 2px solid green;")
            self.valueChanged.emit()
        elif res in [QValidator.Invalid, QValidator.Intermediate]:
            self.is_valid = False
            self.line_edit.setStyleSheet("border: 2px solid red;")
        #TODO not needed so far (un/polish trick) https://forum.qt.io/topic/120591/change-style-sheet-on-the-fly/10
        #https://8bitscoding.github.io/qt/ui/dynamic-p-and-s/
        #self.line_edit.style().unpolish(self.line_edit)
        #self.line_edit.style().polish(self.line_edit)

#TODO ValidatedSizeInputWidget