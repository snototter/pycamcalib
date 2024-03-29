"""Common/basic UI widgets & functionality."""
from typing import List, Sequence, Tuple
from PySide2.QtCore import QEvent, QLocale, QObject, Qt, Signal, Slot
from PySide2.QtGui import QDoubleValidator, QFontDatabase, QIntValidator, QValidator
from PySide2.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QSizePolicy, QWidget

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
        self.decimals = decimals
        layout = QHBoxLayout()
        self.setLayout(layout)
        if label_text is not None:
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
        self.line_edit.installEventFilter(self)
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
            self._setValue(initial_value)

    def valid(self) -> bool:
        return self.is_valid

    def value(self) -> float:
        if self.is_valid:
            return float(self.line_edit.text())
        else:
            return None
    
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched == self.line_edit:
            if event.type() == QEvent.Wheel:
                # Use mouse wheel to increase/decrease the value
                val = float(self.line_edit.text())
                if val is not None:
                    fac = 10 if event.modifiers() & Qt.ControlModifier else 1
                    val = val + fac * (-1.0 if event.delta() < 0 else +1.0)
                    self._setValue(val)
                return True
            elif event.type() in [QEvent.FocusOut, QEvent.Leave]:
                self.editingFinished.emit()
        return super().eventFilter(watched, event)

    def _setValue(self, value):
        fmt = '{:' + (f'.{self.decimals}' if self.decimals is not None else '') + 'f}'
        text = fmt.format(value)
        self.line_edit.setText(text)
        self._textEdited(text)

    @Slot(str)
    def _textEdited(self, text):
        res, modified_text, _ = self.validator.validate(text, 0)
        if res == QValidator.Acceptable:
            self.is_valid = True
            val = float(modified_text)
            self.line_edit.setStyleSheet("border: 2px solid green;")
        elif res in [QValidator.Invalid, QValidator.Intermediate]:
            self.is_valid = False
            self.line_edit.setStyleSheet("border: 2px solid red;")
        self.valueChanged.emit()


class ValidatedIntegerInputWidget(QWidget):
    editingFinished = Signal()
    valueChanged = Signal()

    def __init__(self, label_text: str, initial_value: int = None,
                 min_val: int = None, max_val: int = None,
                 divisible_by: int = None, division_remainder: int = 0,
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.initial_value = initial_value
        self.setLayout(layout)
        if label_text is not None:
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
        self.line_edit.installEventFilter(self)
        self.line_edit.editingFinished.connect(self.editingFinished)
        layout.addWidget(self.line_edit)
        self.is_valid = True
        self.divisible_by = divisible_by
        self.division_remainder = division_remainder
        self.validator = QIntValidator()
        if min_val is not None:
            self.validator.setBottom(min_val)
        if max_val is not None:
            self.validator.setTop(max_val)
        self.line_edit.setValidator(self.validator)
        self.line_edit.textEdited.connect(self._textEdited)
        if initial_value is not None:
            self._setValue(initial_value)
    
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched == self.line_edit:
            if event.type() == QEvent.Wheel:
                # Use mouse wheel to increase/decrease the value
                self.line_edit.setFocus(Qt.MouseFocusReason)
                val = int(self.line_edit.text())
                if val is not None:
                    fac = 10 if event.modifiers() & Qt.ControlModifier else 1
                    val = val + fac * (-1 if event.delta() < 0 else +1)
                    self._setValue(val)
                return True
            elif event.type() in [QEvent.FocusOut, QEvent.Leave]:
                self.editingFinished.emit()
        return super().eventFilter(watched, event)

    def valid(self) -> bool:
        return self.is_valid

    def value(self) -> int:
        if self.is_valid:
            return int(self.line_edit.text())
        else:
            return None

    def _setValue(self, value: int) -> None:
        text = str(value)
        self.line_edit.setText(text)
        self._textEdited(text)

    @Slot(str)
    def _textEdited(self, text):
        res, modified_text, _ = self.validator.validate(text, 0)
        # print(f'Validated "{text}": {res}', self.validator.bottom(), self.validator.top())

        if res == QValidator.Acceptable:
            val = int(modified_text)
            # Check if the user wants the number to be divisible by X, giving
            # some specified remainder
            if self.divisible_by is not None and val % self.divisible_by != self.division_remainder:
                res = QValidator.Invalid
            else:
                self.is_valid = True
                self.line_edit.setStyleSheet("border: 2px solid green;")
                # self.valueChanged.emit()
        if res in [QValidator.Invalid, QValidator.Intermediate]:
            self.is_valid = False
            self.line_edit.setStyleSheet("border: 2px solid red;")
        self.valueChanged.emit()


class SelectionInputWidget(QWidget):
    valueChanged = Signal()

    def __init__(self, label_text: str, choices: Sequence[Tuple[int, str]], initial_idx: int = 0,
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        if label_text is not None:
            # Label to the left:
            lbl = QLabel(label_text)
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            lbl.setAlignment(Qt.AlignLeft)
            layout.addWidget(lbl)
        # Input to the right
        self.selection = QComboBox()
        self.selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for cid, ctxt in choices:
            self.selection.addItem(ctxt, cid)
        # Select initial element
        if initial_idx is not None:
            self.selection.setCurrentIndex(initial_idx)
        self.selection.currentIndexChanged.connect(lambda _: self.valueChanged.emit())
        layout.addWidget(self.selection)

    def valid(self) -> bool:
        return True

    def value(self) -> Tuple[int, str]:
        return (self.selection.currentData(), self.selection.currentText())

    @Slot(str)
    def _textEdited(self, text):
        res, modified_text, _ = self.validator.validate(text, 0)
        if res == QValidator.Acceptable:
            self.is_valid = True
            val = int(modified_text)
            self.line_edit.setStyleSheet("border: 2px solid green;")
            self.valueChanged.emit()
        elif res in [QValidator.Invalid, QValidator.Intermediate]:
            self.is_valid = False
            self.line_edit.setStyleSheet("border: 2px solid red;")


class ValidatedSizeInputWidget(QWidget):
    editingFinished = Signal()
    valueChanged = Signal()

    def __init__(self, label_text: str, initial_value: Tuple[int, int] = (None, None),
                 min_val: Tuple[int, int] = (None, None), max_val: Tuple[int, int] = (None, None),
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        # Label to the left:
        lbl = QLabel(label_text)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl.setAlignment(Qt.AlignLeft)
        layout.addWidget(lbl)
        # Inputs to the right
        self.rows = ValidatedIntegerInputWidget(None, initial_value[0], min_val[0], max_val[0])
        self.rows.valueChanged.connect(self.valueChanged)
        self.rows.editingFinished.connect(self.editingFinished)
        layout.addWidget(self.rows)
        lblx = QLabel('x')
        layout.addWidget(lblx)
        self.columns = ValidatedIntegerInputWidget(None, initial_value[1], min_val[1], max_val[1])
        self.columns.valueChanged.connect(self.valueChanged)
        self.columns.editingFinished.connect(self.editingFinished)
        layout.addWidget(self.columns)

    def valid(self) -> bool:
        return self.rows.is_valid and self.columns.is_valid

    def value(self) -> Tuple[int, int]:
        if self.valid():
            return (self.rows.value(), self.columns.value())
        else:
            return (None, None)
