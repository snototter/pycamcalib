import os
from PySide2.QtCore import Qt, Slot
from .image_view import ImageViewer
from PySide2.QtWidgets import QComboBox, QGridLayout, QSizePolicy, QVBoxLayout, QWidget

from ...processing import ImageSource, Preprocessor

#TODO 03.10. layout preview (top alignment??)
#TODO 04.10. add step slider


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
        # If we only scale the image within the viewer upon construction, it 
        # might show at a wrong scale (because the window may be resized 
        # afterwards, e.g. when we show it in an PreproOperation's config
        # dialog). Thus, we'll scale to fit upon the first resizeEvent
        self._delay_preview_scale_to_fit = True
        self._updateImageList()
    
    def _initLayout(self):
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)

        self.combo_image_selection = QComboBox()
        self.combo_image_selection.setMinimumHeight(26)
        self.combo_image_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_image_selection.currentIndexChanged.connect(self._onImageSelectionChanged)
        layout_main.addWidget(self.combo_image_selection)

        if self.with_step_slider:
            print('TODO TODO TODO step selection slider still missing!')

        self.preview = ImageViewer()
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout_main.addWidget(self.preview)

        
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)

    
    def _updateImageList(self):
        self.combo_image_selection.clear()
        if self.image_source is not None:
            for filename in self.image_source.filenames():
                self.combo_image_selection.addItem(os.path.basename(filename))
        if self.combo_image_selection.count() > 0:
            self.combo_image_selection.setCurrentIndex(0)

    def resizeEvent(self, event):
        if self._delay_preview_scale_to_fit and self.combo_image_selection.count() > 0:
            # At the first resizeEvent after construction, we need to scale the
            # image viewer to properly "fit to window" (if we already have an
            # image selected)
            self._delay_preview_scale_to_fit = False
            self.preview.scaleToFitWindow()

    @Slot(int)
    def _onImageSelectionChanged(self, index):
        image = self.image_source[index]
        if self.preprocessor is not None:
            image = self.preprocessor.apply(image, self.current_step)
        self.preview.showImage(image, reset_scale=True)
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
        if self.image_source is not None:
            self._onImageSelectionChanged(self.combo_image_selection.currentIndex())
