"""Configuration widgets for supported preprocessing operations."""
import inspect
import sys
from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from pcc.ui.widgets.preprocessing_preview import Previewer, ImageComboboxWidget

from ...processing import PreProcOpGammaCorrection, PreProcOpCLAHE
from .common import displayError, ValidatedFloatInputWidget
from .image_view import ImageViewer


#TODO add CLAHEConfigWidget

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
        gb_config.setFixedWidth(200)

        # 2nd column: preview
        gb_preview = QGroupBox("Preview")
        gb_preview.setLayout(QVBoxLayout())
        layout_row.addWidget(gb_preview)
        
        image_selection = ImageComboboxWidget(self.image_source)
        gb_preview.layout().addWidget(image_selection)
        self.preview = Previewer(self.preprocessor, self.preproc_step + 1)
        image_selection.imageSelectionChanged.connect(self.preview.onImageSelectionChanged)
        # Populate current image
        self.preview.onImageSelectionChanged(*image_selection.getImageSelection())

        gb_preview.layout().addWidget(self.preview)

        # Accept/reject buttons at the bottom
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout_main.addWidget(btn_box)

    @Slot()
    def onConfigurationUpdated(self):
        self.preview.onPreprocessorChanged(self.preprocessor)
