import os
from PySide2.QtCore import Slot
from .image_view import ImageViewer
from PySide2.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ...processing import ImageSource, Preprocessor

class Previewer(QWidget):
    def __init__(self, image_source: ImageSource, preprocessor: Preprocessor,
                 with_step_slider: bool = False, current_step: int = -1, parent=None):
        """
        with_step_slider:   let the user select up until which operation the
                            preprocessing pipeline should be applied
        
        current_step:       up until which operation the preproc. pipeline should
                            be applied if with_step_slider is False. Otherwise,
                            this defines the range slider's initial value
        """
        super().__init__(parent)
        self.image_source = image_source
        self.preprocessor = preprocessor
        self.with_step_slider = with_step_slider
        self.current_step = current_step
        self._initLayout()
        self._updateImageList()
    
    def _initLayout(self):
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)

        self.combo_image_selection = QComboBox()
        self.combo_image_selection.currentIndexChanged.connect(self._onImageSelectionChanged)
        layout_main.addWidget(self.combo_image_selection)

        if self.with_step_slider:
            print('TODO TODO TODO step selection slider still missing!')

        self.preview = ImageViewer()
        layout_main.addWidget(self.preview)
        
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)
    
    def _updateImageList(self):
        self.combo_image_selection.clear()
        if self.image_source is not None:
            for filename in self.image_source.filenames():
                self.combo_image_selection.addItem(os.path.basename(filename))
        if self.combo_image_selection.count() > 0:
            self.combo_image_selection.setCurrentIndex(0)

    @Slot(int)
    def _onImageSelectionChanged(self, index):
        self.preview.showImage(self.image_source[index], reset_scale=True)
        self.preview.scaleToFitWindow()
    
    @Slot(ImageSource)
    def onImageSourceChanged(self, image_source: ImageSource):
        self.image_source = image_source
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)
        self._updateImageList()

    @Slot(Preprocessor)
    def onPreprocessorChanged(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)
