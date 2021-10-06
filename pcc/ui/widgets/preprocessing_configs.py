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

    configurationUpdated = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.gamma_widget = ValidatedFloatInputWidget('Gamma:', self.operation.gamma, 0.01, 30, 2)
        self.gamma_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.gamma_widget)

        button = QPushButton('Apply')
        button.clicked.connect(self._updateParameters)
        layout_main.addWidget(button)

        layout_main.addStretch()
    
    @Slot()
    def _updateParameters(self):
        if self.gamma_widget.valid():
            self.operation.set_gamma(self.gamma_widget.value())
            self.configurationUpdated.emit()
        else:
            displayError('Configuration is invalid, please validate gamma.', parent=self)


class CLAHEConfigWidget(QWidget):
    operation_name = PreProcOpCLAHE.name

    configurationUpdated = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.clip_widget = ValidatedFloatInputWidget('Clip Limit:', self.operation.clip_limit, decimals=1)
        self.clip_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.clip_widget)

        self.tile_widget = ValidatedSizeInputWidget('Tile Size:', self.operation.tile_size, (1, 1))
        self.tile_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.tile_widget)

        button = QPushButton('Apply')
        button.clicked.connect(self._updateParameters)
        layout_main.addWidget(button)

        layout_main.addStretch()
    
    @Slot()
    def _updateParameters(self):
        if self.clip_widget.valid() and self.tile_widget.valid():
            self.operation.set_clip_limit(self.clip_widget.value())
            self.operation.set_tile_size(self.tile_widget.value())
            self.configurationUpdated.emit()
        else:
            displayError('Configuration is invalid, please change the parameters.', parent=self)


class ThresholdConfigWidget(QWidget):
    operation_name = PreProcOpThreshold.name

    configurationUpdated = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.threshold_widget = ValidatedIntegerInputWidget('Threshold:', self.operation.threshold_value, 0, 255)
        self.threshold_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.threshold_widget)

        self.max_widget = ValidatedIntegerInputWidget('Max. Value:', self.operation.max_value, 0, 255)
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
        self.type_widget.valueChanged.connect(self._updateParameters)
        layout_main.addWidget(self.type_widget)

        button = QPushButton('Apply')
        button.clicked.connect(self._updateParameters)
        layout_main.addWidget(button)

        layout_main.addStretch()
    
    @Slot()
    def _updateParameters(self):
        if self.threshold_widget.valid() and self.max_widget.valid():
            self.operation.set_threshold_value(self.threshold_widget.value())
            self.operation.set_max_value(self.max_widget.value())
            self.operation.set_threshold_type(self.type_widget.value()[0])
            self.configurationUpdated.emit()
        else:
            displayError('Configuration is invalid, please change the parameters.', parent=self)


class AdaptiveThresholdConfigWidget(QWidget):
    operation_name = PreProcOpAdaptiveThreshold.name

    configurationUpdated = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        
        self.max_widget = ValidatedIntegerInputWidget('Max. Value:', self.operation.max_value, 0, 255)
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
        self.type_widget.valueChanged.connect(self._updateParameters)
        layout_main.addWidget(self.type_widget)

        # Block size must be odd and > 0
        self.block_size_widget = ValidatedIntegerInputWidget('Block size:', self.operation.block_size, 1,
                                                             divisible_by=2, division_remainder=1)
        self.block_size_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.block_size_widget)

        self.cval_widget = ValidatedFloatInputWidget('Constant:', self.operation.C, decimals=1)
        self.cval_widget.editingFinished.connect(self._updateParameters)
        layout_main.addWidget(self.cval_widget)

        button = QPushButton('Apply')
        button.clicked.connect(self._updateParameters)
        layout_main.addWidget(button)

        layout_main.addStretch()
    
    @Slot()
    def _updateParameters(self):
        if self.max_widget.valid() and self.block_size_widget.valid() and self.cval_widget.valid():
            self.operation.set_max_value(self.max_widget.value())
            self.operation.set_threshold_type(self.type_widget.value()[0])
            self.operation.set_method(self.method_widget.value()[0])
            self.operation.set_block_size(self.block_size_widget.value())
            self.operation.set_C(self.cval_widget.value())
            self.configurationUpdated.emit()
        else:
            displayError('Configuration is invalid, please change the parameters.', parent=self)


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
        config_widget = wclass(self.operation)
        gb_config.layout().addWidget(config_widget)
        config_widget.configurationUpdated.connect(self.onConfigurationUpdated)
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
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout_main.addWidget(btn_box)
        self.setMinimumWidth(600)

    def setSelectedPreviewIndex(self, index: int) -> None:
        self.image_selection.setCurrentIndex(index)
        sel = self.image_selection.getImageSelection()
        # Setting the combobox's index programmatically doesn't trigger the signal
        self.preview.onImageSelectionChanged(*self.image_selection.getImageSelection())

    @Slot()
    def onConfigurationUpdated(self):
        self.preview.onPreprocessorChanged(self.preprocessor)
