import os
from typing import Tuple
from PySide2.QtCore import QEvent, QObject, Qt, Signal, Slot
from PySide2.QtGui import QWheelEvent
import numpy as np
from .image_view import ImageViewer
from PySide2.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget

from ...processing import ImageSource, Preprocessor

class SliderWidget(QWidget):
    """Adapted from iminspect.inputs"""
    # Emitted whenever the slider's value changes (either by the user or programmatically)
    valueChanged = Signal(object)

    def __init__(self, label, min_value=0, max_value=100, num_steps=10,
            initial_value=None,
            value_convert_fx=lambda v: int(v), # Type conversion (internally this slider uses floats)
            value_format_fx=lambda v: f'{v:4d}', # Maps slider value => string
            min_label_width=None, value_display_left=False, parent=None):
        super().__init__(parent)
        self._min_value = min_value
        self._max_value = max_value
        self._num_steps = num_steps
        self._step_size = (max_value - min_value) / num_steps
        self._value_format_fx = value_format_fx
        self._value_convert_fx = value_convert_fx

        layout = QHBoxLayout()
        lbl = QLabel(label)
        if min_label_width is not None:
            lbl.setMinimumWidth(min_label_width)
        layout.addWidget(lbl)

        self._slider_label = QLabel(' ')
        if value_display_left:
            layout.addWidget(self._slider_label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(num_steps)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.valueChanged.connect(self._onValueChanged)
        self._slider.installEventFilter(self) # We want to override the default mouse wheel behavior too
        layout.addWidget(self._slider)

        if not value_display_left:
            layout.addWidget(self._slider_label)

        # Set label to maximum value, so we can fix the width
        self._slider_label.setText(value_format_fx(max_value))
        self._slider_label.setFixedWidth(self._slider_label.sizeHint().width())

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        if initial_value is None:
            self._slider.setValue(self._toSliderValue(min_value))
        else:
            self._slider.setValue(self._toSliderValue(initial_value))
        self._onValueChanged()

    def _toSliderValue(self, value):
        v = round((value - self._min_value)/self._step_size)
        return v

    def value(self):
        return self._sliderValue()

    def _sliderValue(self):
        v = self._slider.value()
        v = self._min_value + v * self._step_size
        return self._value_convert_fx(v)

    def _onValueChanged(self):
        val = self._sliderValue()
        self._slider_label.setText(self._value_format_fx(val))
        self.valueChanged.emit(val)

    def setValue(self, v):
        self._slider.setValue(self._toSliderValue(v))
        self._onValueChanged()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Wheel and watched == self._slider:
            # Discard mouse wheel for the slider (because we - its parent - will
            # receive a wheelEvent, too)
            return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.isEnabled():
            return
        fac = -1 if event.delta() < 0 else +1
        self._slider.setValue(self._toSliderValue(self.value() + fac * self._step_size))


class PreProcStepSliderWidget(SliderWidget):
    def __init__(self, image_source, preprocessor, label, min_value=0, max_value=100, num_steps=10,
                 initial_value=None,
                 value_convert_fx=lambda v: int(v), # Type conversion (internally this slider uses floats)
                 value_format_fx=lambda v: f'{v:4d}', # Maps slider value => string
                 min_label_width=None, value_display_left=True, parent=None):
        super().__init__(label, min_value, max_value, num_steps, initial_value,
                         value_convert_fx, value_format_fx, min_label_width,
                         value_display_left, parent)
        self.image_source = None
        self.preprocessor = None
        self.onImageSourceChanged(image_source)
        self.onPreprocessorChanged(preprocessor)
    
    @Slot(ImageSource)
    def onImageSourceChanged(self, image_source: ImageSource) -> None:
        self.image_source = image_source
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)

    @Slot(Preprocessor)
    def onPreprocessorChanged(self, preprocessor: Preprocessor) -> None:
        self.preprocessor = preprocessor
        self.setEnabled(self.image_source is not None and self.preprocessor is not None)
        if self.preprocessor is not None:
            # Adjust range (change step to max number of steps if it was at max previously, too)
            self._max_value = self.preprocessor.num_operations()
            self._num_steps = self._max_value - 1
            self._step_size = (self._max_value - 1) / self._num_steps
            self._slider.setMinimum(0)
            self._slider.setMaximum(self._num_steps)
            self._slider.setValue(self._num_steps)


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
        layout_main.setContentsMargins(0, 0, 0, 0)
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
