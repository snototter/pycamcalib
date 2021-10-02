"""Configuration widgets for supported preprocessing operations."""
import inspect
import sys
from PySide2.QtCore import QLocale, Qt, Signal, Slot
from PySide2.QtGui import QDoubleValidator, QFontDatabase, QValidator
from PySide2.QtWidgets import QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QVBoxLayout, QWidget

from ...processing import PreProcOpGammaCorrection, PreProcOpCLAHE


## Removed because we want a custom ordering within the UI combobox
# def generate_operation_mapping():
#     """Returns a 'name':class mapping for all preprocessing operations
#     defined within this module."""
#     operations = dict()
#     for name, obj in inspect.getmembers(sys.modules[__name__]):
#         if inspect.isclass(obj) and name.startswith('PreProcOp'):
#             operations[obj.name] = obj
#     return operations


class InputFloat(QWidget):
    editingFinished = Signal(float)
    valueChanged = Signal(float)

    def __init__(self, text, min_val, max_val, decimals=0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.setLayout(layout)
        # Label to the left:
        lbl = QLabel(text)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl.setAlignment(Qt.AlignLeft)
        layout.addWidget(lbl)
        # Input to the right
        self.line_edit = QLineEdit()
        # self.line_edit.setInputMask('000.000.000.000;_')
        self.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.line_edit.setAlignment(Qt.AlignRight)
        # self.line_edit.editingFinished.connect(lambda: self.editingFinished.emit(float(self.line_edit.text())))
        layout.addWidget(self.line_edit)
        # Add input validation to allow only floating point numbers. 
        # Additionally, force '.' as decimal point (instead of comma) by
        # overriding the validator's locale:
        locale = QLocale(QLocale.C)
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        self.validator = QDoubleValidator()
        self.validator.setLocale(locale)
        self.validator.setRange(min_val, max_val, decimals)
        self.line_edit.setValidator(self.validator)
        self.line_edit.textEdited.connect(self._textEdited)

    @Slot(str)
    def _textEdited(self, text):
        res, modified_text, pos = self.validator.validate(text, 0) #TODO > top yield INTERMEDIATE
        print(f'Validated "{text}": {res}')
        # if res == QValidator.Intermediate:
            # # QDoubleValidator interprets out-of-range inputs also as "intermediate"
            # try:
            #     val = float(modified_text)
            #     if val < self.validator.bottom() or val > self.validator.top():
            #         res = QValidator.Invalid
            #     else:
            #         self.line_edit.setStyleSheet("border: none;")
            # except ValueError:
            #     res = QValidator.Invalid

        if res == QValidator.Acceptable:
            val = float(modified_text)
            self.valueChanged.emit(val)
            self.line_edit.setStyleSheet("border: 3px solid green;")
            
        elif res in [QValidator.Invalid, QValidator.Intermediate]:
            self.line_edit.setStyleSheet("border: 3px solid red;")
        #TODO not needed so far (un/polish trick) https://forum.qt.io/topic/120591/change-style-sheet-on-the-fly/10
        #https://8bitscoding.github.io/qt/ui/dynamic-p-and-s/
        #self.line_edit.style().unpolish(self.line_edit)
        #self.line_edit.style().polish(self.line_edit)


class GammaCorrectionConfigWidget(QWidget):
    operation_name = PreProcOpGammaCorrection.name

    configurationUpdated = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        #TODO remove
        # self.operation.set_gamma(3)
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        gamma_widget = InputFloat('Gamma:', 0.01, 30, 3)
        gamma_widget.editingFinished.connect(self._updateGamma)#FIXME check if editingFinished works sufficiently well (or do we need a button to manually set it?)
        # gamma_widget.valueChanged.connect(self._updateGamma) #FIXME computationally too expensive (will update preview upon every acceptable input...)
        
        layout_main.addWidget(gamma_widget)
        layout_main.addStretch()
    
    @Slot(float)
    def _updateGamma(self, gamma):
        self.operation.set_gamma(gamma)
        self.configurationUpdated.emit()


class PreProcOpConfigDialog(QDialog):
    """A dialog to configure preprocessing operations.

This will show a two-column dialog, with configuration options to the left and
image preview to the right.
In order to enable the image preview, the input_image must be given (i.e. the
image which will be passed as input to the operation's 'apply()' method)."""
    def __init__(self, operation, input_image=None, parent=None):
        super().__init__(parent)
        self.input_image = input_image
        self.operation = operation
        self.initial_config = operation.freeze()

        self._generateConfigWidgetMapping()
        self._initLayout()
        self.setWindowTitle(f'Configure {operation.display}')
        self.setModal(True)

    def restoreConfiguration(self):
        """To be invoked by caller if the user cancelled this dialog."""
        self.operation.configure(self.initial_config)

    def _generateConfigWidgetMapping(self):
        # Create a mapping from PreProcOp.name (str) to its configuration widget
        self._config_widget_mapping = dict()
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and name.endswith('ConfigWidget'):
                self._config_widget_mapping[obj.operation_name] = obj

    def _initLayout(self):
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        layout_row = QHBoxLayout()
        layout_main.addLayout(layout_row)
        # 1st column: operation-specific configuration widget
        gb_config = QGroupBox("Settings")
        gb_config.setLayout(QVBoxLayout())
        layout_row.addWidget(gb_config)

        wclass = self._config_widget_mapping[self.operation.name]
        config_widget = wclass(self.operation)
        gb_config.layout().addWidget(config_widget)
        # config_widget.configurationUpdated.connect(update preview!) #TODO update preview

        # 2nd column: preview
        gb_preview = QGroupBox("Preview")
        gb_preview.setLayout(QVBoxLayout())
        layout_row.addWidget(gb_preview)
        #TODO remove
        from .gallery import PlaceholderWidget
        self.preview = PlaceholderWidget()
        gb_preview.layout().addWidget(self.preview)

        # Accept/reject buttons at the bottom
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel) #TODO try apply
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout_main.addWidget(btn_box)
