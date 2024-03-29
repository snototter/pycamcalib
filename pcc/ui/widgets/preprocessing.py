import inspect
import logging
import os
import pathlib
import numpy as np
from PySide2.QtCore import QSize, Qt, Signal, Slot
from PySide2.QtGui import QIcon, QPalette
from PySide2.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QSpacerItem, QToolButton, QVBoxLayout, QWidget

from ...processing import ImageSource, ConfigurationError, Preprocessor, PreProcOpGrayscale, PreProcOperationBase, AVAILABLE_PREPROCESSOR_OPERATIONS
from .common import HorizontalLine, displayError, ignoreMessageCallback
from .preprocessing_configs import PreProcOpConfigDialog
from .preprocessing_preview import Previewer, ImageComboboxWidget, PreProcStepSliderWidget


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
    moveUp = Signal(int)
    moveDown = Signal(int)
    toggled = Signal(int, bool)
    remove = Signal(int)
    configurationChanged = Signal(int)

    def __init__(self, image_source: ImageSource, preprocessor: Preprocessor,
                 operation: PreProcOperationBase, list_index: int,
                 number_operations: int, preview_image_index: int, parent=None):
        super().__init__(parent)
        self.image_source = image_source
        self.preview_image_index = preview_image_index
        self.preprocessor = preprocessor
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
            btn_config.clicked.connect(self.onConfigure)
            layout.addWidget(btn_config)

        # Allow the user to quickly enable/disable a single preprocessing step
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(operation.enabled)
        self.checkbox.stateChanged.connect(lambda state: self.toggled.emit(self.list_index, state==Qt.Checked))
        layout.addWidget(self.checkbox)

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
        # if self.list_index % 2 == 1:
        #     self.setStyleSheet("background-color: red;")

    @Slot()
    def onConfigure(self):
        # Ensure that the operation is enabled
        initially_enabled = self.operation.enabled
        self.operation.set_enabled(True)
        self.checkbox.setChecked(self.operation.enabled)
        # Show configuration dialog
        dlg = PreProcOpConfigDialog(self.list_index, self.image_source,
                                    self.preprocessor, self.operation, self)
        dlg.setSelectedPreviewIndex(self.preview_image_index)
        if dlg.exec_():
            # Check if the parameters actually differ:
            if dlg.hasConfigurationChanged():
                self.configurationChanged.emit(self.list_index)
        else:
            dlg.restoreConfiguration()
            self.operation.set_enabled(initially_enabled)

    def update(self):
        self.label.setText(self.operation.description())
        return super().update()

    @Slot(ImageSource)
    def onImageSourceChanged(self, image_source: ImageSource):
        self.image_source = image_source

    @Slot(Preprocessor)
    def onPreprocessorChanged(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
    
    @Slot(int, np.ndarray)
    def onImageSelectionChanged(self, index: int, image: np.ndarray):
        self.preview_image_index = index


class AddOperationItem(QWidget):
    """Special list item which allows adding another preprocessing operation"""
    # Emits the selected operations' index into AVAILABLE_PREPROCESSOR_OPERATIONS
    addOperation = Signal(PreProcOperationBase)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout_main = QVBoxLayout()
        # Add a horizontal divider between configured operations and this
        # special item
        layout_main.addWidget(HorizontalLine())
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
    """Two-column widget to configure & preview the preprocessing pipeline."""
    preprocessorChanged = Signal(Preprocessor)
    _imageSourceChanged = Signal(ImageSource)

    def __init__(self, image_source, message_callback=ignoreMessageCallback, icon_size=QSize(20, 20), parent=None):
        super().__init__(parent)
        self.image_source = image_source
        # We always start with grayscale conversion
        self.preprocessor = Preprocessor()
        self.preprocessor.add_operation(PreProcOpGrayscale())
        self.preprocessorChanged.emit(self.preprocessor)
        # Supported file filters for loading/saving:
        self._file_filters = ["All Files (*.*)", "TOML (*.toml)"]
        self._file_filter_preferred_idx = 1
        # Relevant status messages will be propagated to the parent's status
        # bar via the specified message callback
        self.showMessage = message_callback

        # When saving the pipeline, we suggest the user the same directory
        # a config has been loaded from (if s/he opened a config).
        # Empty string results in the current/last opened directory (qt default)
        self._previously_selected_folder = ''

        # Set up UI
        self._initLayout(icon_size)
        self._updateList()

    def _initLayout(self, icon_size: QSize) -> None:
        layout_main = QGridLayout()
        self.setLayout(layout_main)
        ## 1st row: load/save + image selection
        layout_controls = QHBoxLayout()
        layout_main.addLayout(layout_controls, 0, 0, 1, 1)

        btn_load = QPushButton(' Load')
        btn_load.setIcon(QIcon.fromTheme('document-open'))
        btn_load.setIconSize(icon_size)
        btn_load.setToolTip('Load Preprocessing Pipeline')
        btn_load.setMinimumHeight(icon_size.height() + 6)
        btn_load.clicked.connect(self.onLoadPipeline)
        layout_controls.addWidget(btn_load)

        self._btn_save = QPushButton(' Save')
        self._btn_save.setIcon(QIcon.fromTheme('document-save'))
        self._btn_save.setIconSize(icon_size)
        self._btn_save.setToolTip('Save Preprocessing Pipeline')
        self._btn_save.setMinimumHeight(icon_size.height() + 6)
        self._btn_save.clicked.connect(self.onSavePipeline)
        layout_controls.addWidget(self._btn_save)

        # Image selection
        self.image_selection = ImageComboboxWidget(self.image_source)
        self._imageSourceChanged.connect(self.image_selection.onImageSourceChanged)
        layout_main.addWidget(self.image_selection, 0, 1, 1, 1)

        # 2nd row contains the list widget + step slider
        self.list_widget = QListWidget()
        # Disable selection/highlighting:
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        palette = QPalette()
        palette.setColor(QPalette.Highlight, self.list_widget.palette().color(QPalette.Base))
        palette.setColor(QPalette.HighlightedText, self.list_widget.palette().color(QPalette.Text))
        self.list_widget.setPalette(palette)
        # # self.list_widget.setAlternatingRowColors(True) #Doesn't work
        # # self.list_widget.setStyleSheet("alternate-background-color: white; background-color: blue;")
        layout_main.addWidget(self.list_widget, 1, 0, 2, 1)

        self.step_slider = PreProcStepSliderWidget(self.image_source, self.preprocessor,
                                                  'Preproc. Steps:', 1, self.preprocessor.num_operations(),
                                                  self.preprocessor.num_operations()-1)
        self._imageSourceChanged.connect(self.step_slider.onImageSourceChanged)
        self.preprocessorChanged.connect(self.step_slider.onPreprocessorChanged)
        layout_main.addWidget(self.step_slider, 1, 1, 1, 1)

        # Preview
        self.preview = Previewer(self.preprocessor, -1)
        self.preprocessorChanged.connect(self.preview.onPreprocessorChanged)
        self.image_selection.imageSelectionChanged.connect(self.preview.onImageSelectionChanged)
        self.step_slider.valueChanged.connect(self.preview.onStepChanged)
        layout_main.addWidget(self.preview, 2, 1, 1, 1)
        
        layout_main.setRowStretch(2, 10)
        layout_main.setColumnStretch(0, 3)
        layout_main.setColumnStretch(1, 2)

    def _updateList(self) -> None:
        # Add all currently configured operations
        self.list_widget.clear()
        for idx, op in enumerate(self.preprocessor.operations):
            # Add a default item
            item = QListWidgetItem()
            self.list_widget.addItem(item)
            # Initialize the operation item widget
            item_widget = OperationItem(self.image_source, self.preprocessor,
                                        op, idx, self.preprocessor.num_operations(),
                                        self.image_selection.currentIndex())
            item_widget.configurationChanged.connect(self.onOperationConfigurationHasChanged)
            self.preprocessorChanged.connect(item_widget.onPreprocessorChanged)
            self._imageSourceChanged.connect(item_widget.onImageSourceChanged)
            self.image_selection.imageSelectionChanged.connect(item_widget.onImageSelectionChanged)
            item_widget.moveUp.connect(self.onMoveUp)
            item_widget.moveDown.connect(self.onMoveDown)
            item_widget.toggled.connect(self.onOperationCheckboxToggled)
            item_widget.remove.connect(self.onRemove)
            item.setSizeHint(item_widget.minimumSizeHint())
            self.list_widget.setItemWidget(item, item_widget)

        # Last row is reserved to add another operation to the pipeline
        item = QListWidgetItem()#self.list_widget)
        self.list_widget.addItem(item)
        item_widget = AddOperationItem()
        item_widget.addOperation.connect(self.onAddOperation)
        item.setSizeHint(item_widget.minimumSizeHint())
        self.list_widget.setItemWidget(item, item_widget)

        # Enable/disable save button depending on configured pipeline
        self._btn_save.setEnabled(self.preprocessor.num_operations() > 0)

    @Slot(int)
    def onMoveUp(self, op_idx: int) -> None:
        # User wants to change the order of operations
        self.preprocessor.swap_previous(op_idx)
        self.preprocessorChanged.emit(self.preprocessor)
        # Rebuilding the list is easier (takeItem/insertItem needs further
        # investigation, because of the custom ItemWidget - additionally,
        # we would need to adjust all ItemWidget's indices accordingly...)
        self._updateList()
        # item = self.list_widget.takeItem(op_idx)
        # self.list_widget.insertItem(op_idx-1, item)

    @Slot(int)
    def onMoveDown(self, op_idx: int) -> None:
        # User wants to change the order of operations
        self.preprocessor.swap_next(op_idx)
        self.preprocessorChanged.emit(self.preprocessor)
        self._updateList()

    @Slot(int)
    def onRemove(self, op_idx: int) -> None:
        # User wants to remove an operation
        self.preprocessor.remove(op_idx)
        self.preprocessorChanged.emit(self.preprocessor)
        self._updateList()
        self.step_slider.changeToLastStep()

    @Slot(int, bool)
    def onOperationCheckboxToggled(self, op_idx: int, enabled: bool) -> None:
        # Checkbox enabled/disabled has been toggled
        self.preprocessor.set_enabled(op_idx, enabled)
        self.preprocessorChanged.emit(self.preprocessor)

    @Slot(PreProcOperationBase)
    def onAddOperation(self, operation: PreProcOperationBase) -> None:
        # User wants to add another operation to the pipeline
        self.preprocessor.add_operation(operation)
        self.preprocessorChanged.emit(self.preprocessor)
        self._updateList()
        self.step_slider.changeToLastStep()

    @Slot(int)
    def onOperationConfigurationHasChanged(self, _: int) -> None:
        self.preprocessorChanged.emit(self.preprocessor)
        self._updateList()

    @Slot()
    def onLoadPipeline(self) -> None:
        # Let the user select a TOML file. getOpenFileName also returns the applied file filter
        filename, _ = QFileDialog.getOpenFileName(self, 'Load Preprocessing Pipeline from TOML file',
                                                  self._previously_selected_folder,
                                                  ';;'.join(self._file_filters),
                                                  self._file_filters[self._file_filter_preferred_idx])
        if len(filename) > 0 and pathlib.Path(filename).exists():
            # Store corresponding folder for subsequent open/save directory suggestions
            self._previously_selected_folder = os.path.dirname(filename)
            try:
                self.preprocessor.loadTOML(filename)
                self.preprocessorChanged.emit(self.preprocessor)
                self.showMessage(f'Preprocessing pipeline has been loaded from {filename}', 10000)
            except (FileNotFoundError, ConfigurationError) as e:
                _logger.error("Error while loading TOML preprocessor configuration:", exc_info=e)
                self.showMessage(f'Error while TOML preprocessor configuration ({e.__class__.__name__})', 10000)
                displayError(f"Error while loading TOML preprocessor configuration.",
                             informative_text=f'{e.__class__.__name__}: {str(e)}',
                             parent=self)
            self._updateList()


    @Slot()
    def onSavePipeline(self) -> None:
        # Let the user select the output (TOML) file. getSaveFileName also returns the applied file filter
        filename, _ = QFileDialog.getSaveFileName(self, 'Choose TOML file to save Preprocessing Pipeline',
                                                  self._previously_selected_folder,
                                                  ';;'.join(self._file_filters),
                                                  self._file_filters[self._file_filter_preferred_idx])
        if len(filename) > 0:
            # Append TOML extension if not provided
            if not filename.endswith('.toml'):
                filename += '.toml'
            self.preprocessor.saveTOML(filename)
            self.showMessage(f'Preprocessing pipeline has been saved to {filename}', 10000)

    @Slot(ImageSource)
    def onImageSourceChanged(self, image_source: ImageSource) -> None:
        self.image_source = image_source
        self._imageSourceChanged.emit(image_source)

    @Slot(bool)
    def onPreprocPipelineToggled(self, preproc_enabled: bool) -> None:
        self.preprocessor.set_enabled(None, preproc_enabled)
        self.preprocessorChanged.emit(self.preprocessor)
