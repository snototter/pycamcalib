import inspect
import logging
import os
import pathlib
from PySide2.QtCore import QSize, Qt, Signal, Slot
from PySide2.QtGui import QIcon, QPalette
from PySide2.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QSpacerItem, QToolButton, QVBoxLayout, QWidget

from pcc.processing.preprocessing import PreProcOpCLAHE, PreProcOpGammaCorrection, PreProcOpGrayscale, PreProcOpHistEq #TODO remove
from ...processing import ConfigurationError, Preprocessor, PreProcOpGrayscale, PreProcOperationBase, AVAILABLE_PREPROCESSOR_OPERATIONS
from .common import HLine, displayError

#TODO tasks:
# * load TOML
#   * open file, init preprocessor
#   DONE * preprocessor restore ops
# * save TOML
#   * ops freeze, preprocessor freeze/toml export
#   * select file
# * image display (and combine with image loading - populate first image)
#   lot of work
# * configure
#   medium work (clahe + gamma)

_logger = logging.getLogger('PreprocessingUI')


class NumberLabel(QLabel):
    """Fixed width label to display operation number in list widget."""
    def __init__(self, number: int, parent=None):
        text = '' if number is None else f'#{number}'
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setMinimumWidth(self.fontMetrics().boundingRect("99").width())
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class OperationItem(QWidget):
    """A single row of the list widget corresponding to a configured
    preprocessing operation."""
    #TODO emit config changed
    moveUp = Signal(int)
    moveDown = Signal(int)
    toggled = Signal(int, bool)
    remove = Signal(int)

    def __init__(self, operation, list_index, number_operations, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.list_index = list_index
        layout = QHBoxLayout()
        # Display operation number ("step number/index" in preprocessing pipeline)
        # 1-based numbering seems more user-friendly
        lbl_idx = NumberLabel(list_index + 1)
        layout.addWidget(lbl_idx)

        # Speaking name (including parametrization) is provided via an op's "description"
        self.label = QLabel(operation.description())
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.label)

        # Some operations can be configured
        sig = inspect.signature(operation.__init__)
        is_configurable = len(sig.parameters) > 0
        if is_configurable:
            btn_config = QToolButton()
            btn_config.setText('Configure')
            layout.addWidget(btn_config)

        # Allow the user to quickly enable/disable a single preprocessing step
        cb = QCheckBox()
        cb.setChecked(operation.enabled)
        cb.stateChanged.connect(lambda state: self.toggled.emit(self.list_index, state==Qt.Checked))
        layout.addWidget(cb)

        # Let the user reorder the pipeline with up/down movement
        btn_mv_up = QToolButton()
        btn_mv_up.setText('Up')
        btn_mv_up.clicked.connect(lambda: self.moveUp.emit(self.list_index))
        btn_mv_up.setEnabled(list_index > 0)
        layout.addWidget(btn_mv_up)

        btn_mv_down = QToolButton()
        btn_mv_down.setText('Down')
        btn_mv_down.clicked.connect(lambda: self.moveDown.emit(self.list_index))
        btn_mv_down.setEnabled(list_index < number_operations - 1)
        layout.addWidget(btn_mv_down)

        # Let the user delete a single step
        btn_del = QToolButton()
        btn_del.setText('Remove')
        btn_del.clicked.connect(lambda: self.remove.emit(self.list_index))
        layout.addWidget(btn_del)

        self.setLayout(layout)

    def update(self):
        self.label.setText(self.operation.description())
        return super().update()
#FIXME check if adjusting the parameters is persistent (within the preprocessor's list!)

class AddOperationItem(QWidget):
    """Special list item which allows adding another preprocessing operation"""
    # Emits the selected operations' index into AVAILABLE_PREPROCESSOR_OPERATIONS
    addOperation = Signal(PreProcOperationBase)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout_main = QVBoxLayout()
        # Add a horizontal divider between configured operations and this
        # special item
        layout_main.addWidget(HLine())
        layout_row = QHBoxLayout()
        # Placeholder to align with list items above it
        layout_row.addWidget(NumberLabel(None))
        self.combobox = QComboBox()
        # Let the user choose from a list of available operations
        layout_row.addWidget(self.combobox)
        for opcls in AVAILABLE_PREPROCESSOR_OPERATIONS:
            self.combobox.addItem(opcls.display, opcls)
        # User must confirm their selection
        btn = QToolButton()
        btn.setText("Add")
        btn.clicked.connect(self._triggerAdd)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout_row.addWidget(btn)

        layout_main.addLayout(layout_row)
        self.setLayout(layout_main)

    @Slot()
    def _triggerAdd(self):
        # Operation class is provided as an item's userData field, thus we
        # initialize it (operations must ensure a sane default parametrization
        # during construction)
        opcls = self.combobox.currentData()
        self.addOperation.emit(opcls())


class PreprocessingSelector(QWidget):
    """Two-column widget to configure & preview the preprocessing pipeline.
    TODO if images are available, they must be populated to slot X
    """

    def __init__(self, icon_size=QSize(20, 20), parent=None):
        super().__init__(parent)
        # We always start with grayscale conversion
        self.preprocessor = Preprocessor()
        self.preprocessor.add_operation(PreProcOpGrayscale())
        # Supported file filters for loading/saving:
        self._file_filters = ["All Files (*.*)", "TOML (*.toml)"]
        self._file_filter_preferred_idx = 1

        #TODO remove the rest
        self.preprocessor.add_operation(PreProcOpGammaCorrection())
        self.preprocessor.add_operation(PreProcOpHistEq())
        self.preprocessor.add_operation(PreProcOpCLAHE())

        # For saving the pipeline, we suggest the user the same directory
        # a config has been loaded from (if config will be loaded).
        # Empty string results in the current/last opened directory (qt default)
        self._previously_selected_folder = ''

        # Set up UI
        self._initLayout(icon_size)
        self._updateList()

    def _initLayout(self, icon_size):
        layout_main = QVBoxLayout()
        # 1st row: load/save
        layout_controls = QHBoxLayout()
        layout_main.addLayout(layout_controls)

        btn_load = QPushButton(' Load')
        btn_load.setIcon(QIcon.fromTheme('document-open'))
        btn_load.setIconSize(icon_size)
        btn_load.setToolTip('Load Preprocessing Pipeline')
        btn_load.setMinimumHeight(icon_size.height() + 6)
        btn_load.clicked.connect(self._loadPipeline)
        layout_controls.addWidget(btn_load)

        self._btn_save = QPushButton(' Save')
        self._btn_save.setIcon(QIcon.fromTheme('document-open'))
        self._btn_save.setIconSize(icon_size)
        self._btn_save.setToolTip('Save Preprocessing Pipeline')
        self._btn_save.setMinimumHeight(icon_size.height() + 6)
        self._btn_save.clicked.connect(self._savePipeline)
        layout_controls.addWidget(self._btn_save)

        self.list_widget = QListWidget()
        # Disable selection/highlighting:
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        palette = QPalette()
        palette.setColor(QPalette.Highlight, self.list_widget.palette().color(QPalette.Base))
        palette.setColor(QPalette.HighlightedText, self.list_widget.palette().color(QPalette.Text))
        self.list_widget.setPalette(palette)
        # self.list_widget.setAlternatingRowColors(True) #Doesn't work
        # self.list_widget.setStyleSheet("alternate-background-color: white; background-color: blue;")
        layout_main.addWidget(self.list_widget)
        self.setLayout(layout_main)
    
    def _updateList(self):
        #TODO remove:
        import toml
        print('CHECK FROZEN PREPROC:\n', toml.dumps(self.preprocessor.freeze()))

        self.list_widget.clear()

        for idx, op in enumerate(self.preprocessor.operations):
            # Add a default item
            item = QListWidgetItem()
            self.list_widget.addItem(item)
            # Initialize the operation item widget
            item_widget = OperationItem(op, idx, len(self.preprocessor.operations))
            item_widget.moveUp.connect(self._moveUp)
            item_widget.moveDown.connect(self._moveDown)
            item_widget.toggled.connect(self._operationToggled)
            item_widget.remove.connect(self._remove)
            item.setSizeHint(item_widget.minimumSizeHint())
            self.list_widget.setItemWidget(item, item_widget)

        # Last row is reserved to add another operation to the pipeline
        item = QListWidgetItem()#self.list_widget)
        self.list_widget.addItem(item)
        item_widget = AddOperationItem()
        item_widget.addOperation.connect(self._addOperation)
        item.setSizeHint(item_widget.minimumSizeHint())
        self.list_widget.setItemWidget(item, item_widget)

        # Enable/disable save button depending on configured pipeline
        self._btn_save.setEnabled(len(self.preprocessor.operations) > 0)

    @Slot(int)
    def _moveUp(self, op_idx):
        self.preprocessor.swap_previous(op_idx)
        # Rebuilding the list is easier (takeItem/insertItem needs further
        # investigation, because of the custom ItemWidget - additionally,
        # we would need to adjust all ItemWidget's indices accordingly...)
        self._updateList()
        # item = self.list_widget.takeItem(op_idx)
        # self.list_widget.insertItem(op_idx-1, item)

    @Slot(int)
    def _moveDown(self, op_idx):
        self.preprocessor.swap_next(op_idx)
        self._updateList()

    @Slot(int)
    def _remove(self, op_idx):
        self.preprocessor.remove(op_idx)
        self._updateList()

    @Slot(int, bool)
    def _operationToggled(self, op_idx, enabled):
        self.preprocessor.set_enabled(op_idx, enabled)

    @Slot(PreProcOperationBase)
    def _addOperation(self, operation):
        self.preprocessor.add_operation(operation)
        self._updateList()

    @Slot()
    def _loadPipeline(self):
        # Let the user select a TOML file
        # getOpenFileName also return the applied file filter
        filename, _ = QFileDialog.getOpenFileName(self, 'Load Preprocessing Pipeline from TOML file',
                                                  self._previously_selected_folder,
                                                  ';;'.join(self._file_filters),
                                                  self._file_filters[self._file_filter_preferred_idx])
        if len(filename) > 0 and pathlib.Path(filename).exists():
            # Store corresponding folder for subsequent open/save directory suggestions
            self._previously_selected_folder = os.path.dirname(filename)
            try:
                self.preprocessor.loadTOML(filename)
            except (FileNotFoundError, ConfigurationError) as e:
                _logger.error("Error while loading TOML preprocessor configuration:", exc_info=e)
                displayError(f"Error while loading TOML preprocessor configuration.",
                             informative_text=f'{e.__class__.__name__}: {str(e)}',
                             parent=self)
            self._updateList()


    @Slot()
    def _savePipeline(self):
        pass
        #TODO use prev sel folder
#TODO serialize to verify affected changes
