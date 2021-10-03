import os
from typing import Tuple
from PySide2.QtCore import Qt, Signal, Slot
import numpy as np
from .image_view import ImageViewer
from PySide2.QtWidgets import QComboBox, QGridLayout, QSizePolicy, QVBoxLayout, QWidget

from ...processing import ImageSource, Preprocessor

#TODO 04.10. add step slider

class ImageComboboxWidget(QComboBox):
    """A combobox which lists the currently loaded image filenames and emits
    the user-selected image."""
    imageSelectionChanged = Signal(int, np.ndarray)

    def __init__(self, image_source, parent=None):
        super().__init__(parent)
        self.activated.connect(self._onActivated)
        self.onImageSourceChanged(image_source)

    def getImageSelection(self) -> Tuple[int, np.ndarray]:
        """Returns the currently selected (index, image) or (0, None)"""
        if self.image_source is None:
            return (0, None)
        else:
            return (self.currentIndex(), self.image_source[self.currentIndex()])

    @Slot(int)
    def _onActivated(self, index: int) -> None:
        """Everytime the user changes the selection, this 'loads' the image,
        i.e. emits imageSelectionChanged"""
        self.imageSelectionChanged.emit(*self.getImageSelection())

    @Slot(ImageSource)
    def onImageSourceChanged(self, image_source: ImageSource) -> None:
        """Updates the combobox contents and "loads" the first image (emits the
        imageSelectionChanged signal)"""
        self.image_source = image_source
        self.clear()
        if self.image_source is not None:
            for filename in self.image_source.filenames():
                self.addItem(os.path.basename(filename))
        if self.count() > 0:
            self.setCurrentIndex(0)
        self.imageSelectionChanged.emit(*self.getImageSelection())
        self.setEnabled(self.count() > 0)


class Previewer(QWidget):
    def __init__(self, preprocessor: Preprocessor,
                 current_preproc_step: int = -1, parent=None):
        """
        current_preproc_step: up until which operation the preproc. pipeline should
                            be applied if with_step_slider is False. Otherwise,
                            this defines the range slider's initial value
        """
        super().__init__(parent)
        self.image = None
        self.preprocessor = preprocessor
        self.current_preproc_step = current_preproc_step
        self._initLayout()
        # We need to scale the image view's content upon the first
        # resizeEvent (before that our widget dimensions aren't finalized)
        self._first_resize = True

    def _initLayout(self):
        layout_main = QVBoxLayout()
        self.setLayout(layout_main)
        self.preview = ImageViewer()
        layout_main.addWidget(self.preview)
        self.setEnabled(self.preprocessor is not None)

    def resizeEvent(self, event):
        if self._first_resize:
            # At the first resizeEvent after construction, we need to scale the
            # image viewer to properly "fit to window" (if we already have an
            # image selected)
            self._first_resize = False
            self.preview.scaleToFitWindow()

    def _updatePreview(self, reset_scale):
        image = self.image
        if self.preprocessor is not None:
            image = self.preprocessor.apply(image, self.current_preproc_step)
        if image is not None:
            self.preview.showImage(image, reset_scale=reset_scale)

    @Slot(int, np.ndarray)
    def onImageSelectionChanged(self, index: int, image: np.ndarray):
        self.image = image
        self._updatePreview(True)
        if image is not None:
            self.preview.scaleToFitWindow()

    @Slot(Preprocessor)
    def onPreprocessorChanged(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.setEnabled(self.preprocessor is not None)
        self._updatePreview(False)

    @Slot(int)
    def onStepChanged(self, current_step: int):
        self.current_preproc_step = current_step
        self._updatePreview(False)
