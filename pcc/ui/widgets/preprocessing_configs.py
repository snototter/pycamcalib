"""Configuration widgets for supported preprocessing operations."""
import inspect
import sys
from PySide2.QtWidgets import QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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


class GammaCorrectionConfigWidget(QWidget):
    operation_name = PreProcOpGammaCorrection.name

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        #TODO remove
        self.operation.set_gamma(3)
        self._initLayout()
    
    def _initLayout(self):
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)

        layout_main.addWidget(QLabel('TODO Gamma'))
        layout_main.addStretch()


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
