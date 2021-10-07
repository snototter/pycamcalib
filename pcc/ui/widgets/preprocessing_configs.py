"""Configuration widgets for supported preprocessing operations."""
import inspect
import sys
from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from pcc.ui.widgets.preprocessing_preview import Previewer, ImageComboboxWidget

from ...processing import PreProcOpGammaCorrection, PreProcOpCLAHE, PreProcOpThreshold, PreProcOpAdaptiveThreshold
from .common import ValidatedIntegerInputWidget, displayError, ValidatedFloatInputWidget, ValidatedSizeInputWidget, SelectionInputWidget


class GammaCorrectionConfigWidget(QWidget):
    operation_name = PreProcOpGammaCorrection.name

    # Emitted whenever editing a single paramter has finished
    configurationUpdated = Signal()
    # Invoked upon every (intermediate) change; contains the current validation result
    valueChanged = Signal(bool)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.gamma_widget = ValidatedFloatInputWidget('Gamma:', self.operation.gamma, 0.01, 30, 2)
        self.gamma_widget.valueChanged.connect(self._enableButton)
        self.gamma_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.gamma_widget)

        self.button = QPushButton('Apply')
        self.button.clicked.connect(self._updateParameters)
        layout_main.addWidget(self.button)

        layout_main.addStretch()

    def valid(self):
        return self.gamma_widget.valid()
    
    @Slot()
    def _enableButton(self):
        valid = self.valid()
        self.button.setEnabled(valid)
        self.valueChanged.emit(valid)
    
    @Slot()
    def _updateParameters(self):
        if self.valid():
            self.operation.set_gamma(self.gamma_widget.value())
            self.configurationUpdated.emit()
        self._enableButton()


class CLAHEConfigWidget(QWidget):
    operation_name = PreProcOpCLAHE.name

    # Emitted whenever editing a single paramter has finished
    configurationUpdated = Signal()
    # Invoked upon every (intermediate) change; contains the current validation result
    valueChanged = Signal(bool)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.clip_widget = ValidatedFloatInputWidget('Clip Limit:', self.operation.clip_limit, decimals=1)
        self.clip_widget.valueChanged.connect(self._enableButton)
        self.clip_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.clip_widget)

        self.tile_widget = ValidatedSizeInputWidget('Tile Size:', self.operation.tile_size, (1, 1))
        self.tile_widget.valueChanged.connect(self._enableButton)
        self.tile_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.tile_widget)

        self.button = QPushButton('Apply')
        self.button.clicked.connect(self._updateParameters)
        layout_main.addWidget(self.button)

        layout_main.addStretch()

    def valid(self):
        return self.clip_widget.valid() and self.tile_widget.valid()
    
    @Slot()
    def _enableButton(self):
        valid = self.valid()
        self.button.setEnabled(valid)
        self.valueChanged.emit(valid)

    @Slot()
    def _updateParameters(self):
        if self.valid():
            self.operation.set_clip_limit(self.clip_widget.value())
            self.operation.set_tile_size(self.tile_widget.value())
            self.configurationUpdated.emit()
        self._enableButton()


class ThresholdConfigWidget(QWidget):
    operation_name = PreProcOpThreshold.name

    # Emitted whenever editing a single paramter has finished
    configurationUpdated = Signal()
    # Invoked upon every (intermediate) change; contains the current validation result
    valueChanged = Signal(bool)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.threshold_widget = ValidatedIntegerInputWidget('Threshold:', self.operation.threshold_value, 0, 255)
        self.threshold_widget.valueChanged.connect(self._enableButton)
        self.threshold_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.threshold_widget)

        self.max_widget = ValidatedIntegerInputWidget('Max. Value:', self.operation.max_value, 0, 255)
        self.max_widget.valueChanged.connect(self._enableButton)
        self.max_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.max_widget)

        choices = list()
        initial_selection = 0
        for idx in range(len(PreProcOpThreshold.threshold_types)):
            ttype = PreProcOpThreshold.threshold_types[idx]
            if self.operation.threshold_type == ttype[0]:
                initial_selection = idx
            choices.append(ttype)
        self.type_widget = SelectionInputWidget('Type:', choices, initial_selection)
        self.type_widget.valueChanged.connect(self._enableButton)
        self.type_widget.valueChanged.connect(self._updateParameters)
        layout_main.addWidget(self.type_widget)

        self.button = QPushButton('Apply')
        self.button.clicked.connect(self._updateParameters)
        layout_main.addWidget(self.button)

        layout_main.addStretch()

    def valid(self):
        return self.threshold_widget.valid() and self.max_widget.valid()

    @Slot()
    def _enableButton(self):
        valid = self.valid()
        self.button.setEnabled(valid)
        self.valueChanged.emit(valid)
    
    @Slot()
    def _updateParameters(self):
        if self.valid():
            self.operation.set_threshold_value(self.threshold_widget.value())
            self.operation.set_max_value(self.max_widget.value())
            self.operation.set_threshold_type(self.type_widget.value()[0])
            self.configurationUpdated.emit()
        self._enableButton()


class AdaptiveThresholdConfigWidget(QWidget):
    operation_name = PreProcOpAdaptiveThreshold.name

    # Emitted whenever editing a single paramter has finished
    configurationUpdated = Signal()
    # Invoked upon every (intermediate) change; contains the current validation result
    valueChanged = Signal(bool)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        
        self.max_widget = ValidatedIntegerInputWidget('Max. Value:', self.operation.max_value, 0, 255)
        self.max_widget.valueChanged.connect(self._enableButton)
        self.max_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.max_widget)

        choices = list()
        initial_selection = 0
        for idx in range(len(PreProcOpAdaptiveThreshold.methods)):
            method = PreProcOpAdaptiveThreshold.methods[idx]
            if self.operation.method == method[0]:
                initial_selection = idx
            choices.append(method)
        self.method_widget = SelectionInputWidget('Method:', choices, initial_selection)
        self.method_widget.valueChanged.connect(self._enableButton)
        self.method_widget.valueChanged.connect(self._updateParameters)
        layout_main.addWidget(self.method_widget)

        choices = list()
        initial_selection = 0
        for idx in range(len(PreProcOpAdaptiveThreshold.threshold_types)):
            ttype = PreProcOpAdaptiveThreshold.threshold_types[idx]
            if self.operation.threshold_type == ttype[0]:
                initial_selection = idx
            choices.append(ttype)
        self.type_widget = SelectionInputWidget('Type:', choices, initial_selection)
        self.type_widget.valueChanged.connect(self._enableButton)
        self.type_widget.valueChanged.connect(self._updateParameters)
        layout_main.addWidget(self.type_widget)

        # Block size must be odd and > 1
        self.block_size_widget = ValidatedIntegerInputWidget('Block size:', self.operation.block_size, 3,
                                                             divisible_by=2, division_remainder=1)
        self.block_size_widget.valueChanged.connect(self._enableButton)    
        self.block_size_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.block_size_widget)

        self.cval_widget = ValidatedFloatInputWidget('Constant:', self.operation.C, decimals=1)
        self.cval_widget.valueChanged.connect(self._enableButton)
        self.cval_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.cval_widget)

        self.button = QPushButton('Apply')
        self.button.clicked.connect(lambda: self._updateParameters(True))
        layout_main.addWidget(self.button)

        layout_main.addStretch()

    def valid(self):
        return self.max_widget.valid() and self.block_size_widget.valid() and self.cval_widget.valid()

    @Slot()
    def _enableButton(self):
        valid = self.valid()
        self.button.setEnabled(valid)
        self.valueChanged.emit(valid)
    
    @Slot()
    def _updateParameters(self, display_message=False):
        if self.valid():
            self.operation.set_max_value(self.max_widget.value())
            self.operation.set_threshold_type(self.type_widget.value()[0])
            self.operation.set_method(self.method_widget.value()[0])
            self.operation.set_block_size(self.block_size_widget.value())
            self.operation.set_C(self.cval_widget.value())
            self.configurationUpdated.emit()
        self._enableButton()


class PreProcOpConfigDialog(QDialog):
    """A dialog to configure preprocessing operations.

This will show a two-column dialog, with configuration options to the left and
image preview to the right.
In order to enable the image preview, the input_image must be given (i.e. the
image which will be passed as input to the operation's 'apply()' method)."""
    def __init__(self, preproc_step, image_source, preprocessor, operation, parent=None):
        super().__init__(parent)
        self.preproc_step = preproc_step
        self.image_source = image_source
        self.preprocessor = preprocessor
        self.operation = operation
        self.initial_config = operation.freeze()

        self._generateConfigWidgetMapping()
        self._initLayout()
        self.setWindowTitle(f'Configure {operation.display}')
        self.setModal(True)

    def restoreConfiguration(self):
        """To be invoked by caller if the user cancelled this dialog."""
        self.operation.configure(self.initial_config)

    def hasConfigurationChanged(self):
        """Returns True if the operation's current configuration differs from
        its initial configuration."""
        if self.initial_config == self.operation.freeze():
            return False
        return True

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
        self.config_widget = wclass(self.operation)
        gb_config.layout().addWidget(self.config_widget)
        self.config_widget.valueChanged.connect(self.onValueChanged)
        self.config_widget.configurationUpdated.connect(self.onConfigurationUpdated)
        gb_config.setMinimumWidth(200)

        # 2nd column: preview
        gb_preview = QGroupBox("Preview")
        gb_preview.setLayout(QVBoxLayout())
        layout_row.addWidget(gb_preview)
        
        self.image_selection = ImageComboboxWidget(self.image_source)
        gb_preview.layout().addWidget(self.image_selection)
        self.preview = Previewer(self.preprocessor, self.preproc_step + 1)
        self.image_selection.imageSelectionChanged.connect(self.preview.onImageSelectionChanged)
        # Populate current image
        self.preview.onImageSelectionChanged(*self.image_selection.getImageSelection())

        gb_preview.layout().addWidget(self.preview)

        # Accept/reject buttons at the bottom
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout_main.addWidget(self.btn_box)
        self.setMinimumWidth(600)

    def setSelectedPreviewIndex(self, index: int) -> None:
        self.image_selection.setCurrentIndex(index)
        sel = self.image_selection.getImageSelection()
        # Setting the combobox's index programmatically doesn't trigger the signal
        self.preview.onImageSelectionChanged(*self.image_selection.getImageSelection())

    @Slot()
    def onConfigurationUpdated(self):
        if self.config_widget.valid():
            self.preview.onPreprocessorChanged(self.preprocessor)

    @Slot(bool)
    def onValueChanged(self, valid: bool):
        self.btn_box.button(QDialogButtonBox.Ok).setEnabled(valid)
