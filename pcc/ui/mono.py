"""Starts the monocular calibration GUI."""

import html
import io
import sys
import time
import traceback
from PySide2 import QtWidgets
from PySide2.QtCore import QRect, QSize, QTimer, Qt, Signal, Slot
from PySide2.QtWidgets import QApplication, QDesktopWidget, QGridLayout, QGroupBox, QHBoxLayout, QMainWindow, QProgressBar, QSizePolicy, QSplitter, QStatusBar, QStyleFactory, QToolBar, QVBoxLayout, QWidget
from numpy import disp

from ..processing import ImageSource, NoImageDirectoryError, DirectoryNotFoundError, Preprocessor
from .widgets import displayError, CalibrationInputWidget, PreprocessingSelector
import logging

_logger = logging.getLogger('MonoCalibrationGui')
#TODO consistency:
# * Qt style in UI components
# * slots: on<signal>
# * snake_case in non UI submodules

#TODO check:
# not yet: https://realpython.com/python-menus-toolbars/
# tabs: "input" "results" https://pythonspot.com/pyqt5-tabs/
# dock widget for preprocessing: https://pythonpyqt.com/qdockwidget/
# custom listview item https://stackoverflow.com/questions/948444/qlistview-qlistwidget-with-custom-items-and-custom-item-widgets
# collapsible box https://stackoverflow.com/questions/52615115/how-to-create-collapsible-box-in-pyqt
#  also https://groups.google.com/g/python_inside_maya/c/Y6r8o9zpWfU
#  also https://discourse.techart.online/t/pyqt-collapsible-groupbox-or-layout/1108/2

class MonoCalibrationGui(QMainWindow):
    # Emitted whenever an image source has been selected (or changed)
    #TODO connect to calibration widget
    _imageSourceChanged = Signal(ImageSource)

    # TODO not needed! connect the preprocConfigSelector's signal to calibWidget's slot!
    # _preprocessorChanged = Signal(Preprocessor)

    #TODO add calibrationPatternChanged

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Monocular Calibration')

        self.image_source = None
        self.calibration_pattern = None
        self.preprocessor = None
        initial_window_size = QSize(QDesktopWidget().availableGeometry(self).size() *0.6) #TODO* 0.85)

        # self.createActions()
        # self.createMenuBar()
        # self.createToolbars()
        self._createStatusBar()
        self._initLayout(initial_window_size)
        # toolbar_imgsrc = QToolBar('Image Source', self)
        # self.addToolBar(Qt.TopToolBarArea, toolbar_imgsrc)

        self.resize(initial_window_size)
    
    # def createActions(self):
    #     pass

    # def createMenuBars(self):
    #     pass

    # def createToolbars(self):
    #     pass
    
    def _initLayout(self, initial_window_size: QSize):
        layout_main = QVBoxLayout()
        splitter_main = QSplitter(Qt.Vertical)
        layout_main.addWidget(splitter_main)
        ########### 1st row
        splitter_row1 = QSplitter(Qt.Horizontal)
        splitter_main.addWidget(splitter_row1)
        #### Image selection & pattern specification
        layout_row1 = QGridLayout()
        groupbox_input = QGroupBox("Images && Pattern")
        groupbox_input.setLayout(QVBoxLayout())
        # groupbox_input.setStyleSheet("border: 1px solid gray; border-color: #ff17365d;")
        splitter_row1.addWidget(groupbox_input)

        self.calib_input = CalibrationInputWidget()
        self.calib_input.imageFolderSelected.connect(self.onFolderSelected)
        self.calib_input.patternConfigurationChanged.connect(self.onPatternConfigChanged)
        self.calib_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        groupbox_input.layout().addWidget(self.calib_input)

        #### Preprocessing pipeline
        groupbox_preproc = QGroupBox("Preprocessing")
        groupbox_preproc.setLayout(QVBoxLayout())
        groupbox_preproc.setCheckable(True)
        self.preproc_configurator = PreprocessingSelector(self.image_source, message_callback=self.status_bar.showMessage)
        self.preproc_configurator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        groupbox_preproc.toggled.connect(self.preproc_configurator.onPreprocPipelineToggled)
        self._imageSourceChanged.connect(self.preproc_configurator.onImageSourceChanged)
        groupbox_preproc.layout().addWidget(self.preproc_configurator)
        groupbox_preproc.setChecked(False)
        splitter_row1.addWidget(groupbox_preproc)

        splitter_row1.setSizes([initial_window_size.width() * 0.3, initial_window_size.width() * 0.7])
        
        ########### 2nd row (image gallery)
        #TODO connect _imageSourceChanged, preproc_config.preprocpipelineChanged
        # self.preproc_configurator.preprocessorChanged.connect(self.onPreprocessorChanged)
        self.groupbox_imgview = QGroupBox("Images")
        self.groupbox_imgview.setLayout(QVBoxLayout())
        self.groupbox_imgview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # layout_main.addWidget(self.groupbox_imgview)
        splitter_main.addWidget(self.groupbox_imgview)
        from .widgets.gallery import PlaceholderWidget #TODO remove
        self.groupbox_imgview.layout().addWidget(PlaceholderWidget())
        self.groupbox_imgview.setEnabled(False)

        splitter_main.setSizes([initial_window_size.height() * 0.4, initial_window_size.height() * 0.6])
        # splitter.setHandleWidth(80)
        # splitter.setOpaqueResize(False)
        # splitter.setStyleSheet("background-color: #333;")
        central_widget = QWidget()
        central_widget.setLayout(layout_main)
        self.setCentralWidget(central_widget)

    def _createStatusBar(self):
        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self.progress_bar)

    @Slot(str)
    def onFolderSelected(self, folder):
        _logger.info(f'Loading images from {folder}')
        try:
            self.image_source = ImageSource(folder)
            # Notify observers of new/changed image data
            self.groupbox_imgview.setEnabled(True) #TODO reset, then populate with images

            #TODO remove (progress bar dummy test)
            import time
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage(f'Loading {self.image_source.num_images()} images from {folder}')
            for i in range(self.image_source.num_images()):
                prg = int((i+1) / self.image_source.num_images() * 100)
                self.progress_bar.setValue(prg)
                time.sleep(0.1)
            
            # Hide the progress bar after 5 seconds
            timer = QTimer(self)
            timer.singleShot(5000, lambda: self.progress_bar.setVisible(False))
            self.status_bar.showMessage(f'Loaded {self.image_source.num_images()} images from {folder}', 10000)
        except (DirectoryNotFoundError, NoImageDirectoryError) as e:
            _logger.error("Error while loading image files:", exc_info=e)
            self.status_bar.showMessage(f'Error while loading images from {folder} ({e.__class__.__name__})', 10000)
            displayError(f"Error while loading images.", informative_text=str(e), parent=self)
            self.image_source = None
            self.calib_input.resetImageFolder()
        self._imageSourceChanged.emit(self.image_source)
    
    @Slot()
    def onPatternConfigChanged(self):
        print('TODO pattern config changed')

    @Slot(Preprocessor)
    def onPreprocessorChanged(self, preprocessor: Preprocessor):
        _logger.info('TODO preprocessor changed')
        self.preprocessor = preprocessor


def globalExceptionHook(excType, excValue, tracebackobj):
    """
    Global function to catch unhandled exceptions.
    Taken (with minor modifications and obvious API changes) from
    https://www.riverbankcomputing.com/pipermail/pyqt/2009-May/022961.html
    
    @param excType exception type
    @param excValue exception value
    @param tracebackobj traceback object
    """
    separator = '-' * 80
    timestr = time.strftime("%Y-%m-%d, %H:%M:%S")
    tbinfofile = io.StringIO()
    traceback.print_tb(tracebackobj, None, tbinfofile)
    tbinfofile.seek(0)
    tbinfo = tbinfofile.read()
    errmsg = f'{excType}:\n{str(excValue)}'

    sections = [separator, timestr, separator, errmsg, separator, tbinfo]
    _logger.error('Unhandled exception:\n' + '\n'.join(sections))
    displayError('An unhandled exception occurred. Please report the problem.',
                 title='Unhandled Exception',
                 informative_text='<pre>' + html.escape('\n'.join(sections)) + '</pre>')


sys.excepthook = globalExceptionHook


# theme/style: https://pythonbasics.org/pyqt-style/
def run_mono_calibration():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    # print(QtWidgets.QStyleFactory.keys())
    # app.setStyle('Windows')
    # app.setStyle(QStyleFactory.create("Windows"))
    gui = MonoCalibrationGui()
    gui.show()
    app.exec_()

if __name__ == '__main__':
    run_mono_calibration()
