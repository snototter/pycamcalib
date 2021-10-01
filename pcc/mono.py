"""Starts the monocular calibration GUI."""

#TODO move stereo/mono to .ui
from PySide2 import QtWidgets
from PySide2.QtCore import Qt, Slot
from PySide2.QtWidgets import QApplication, QDesktopWidget, QGridLayout, QGroupBox, QHBoxLayout, QMainWindow, QProgressBar, QSizePolicy, QSplitter, QStatusBar, QStyleFactory, QToolBar, QVBoxLayout, QWidget
import sys

from numpy.lib.shape_base import split
from .processing import ImageDirectorySource, NoImageDirectoryError, DirectoryNotFoundError
from .ui.widgets import displayError, CalibrationInputWidget, PreprocessingSelector
import logging

_logger = logging.getLogger('MonoCalibrationGui')


#TODO check:
# not yet: https://realpython.com/python-menus-toolbars/
# tabs: "input" "results" https://pythonspot.com/pyqt5-tabs/
# dock widget for preprocessing: https://pythonpyqt.com/qdockwidget/
# custom listview item https://stackoverflow.com/questions/948444/qlistview-qlistwidget-with-custom-items-and-custom-item-widgets
# collapsible box https://stackoverflow.com/questions/52615115/how-to-create-collapsible-box-in-pyqt
#  also https://groups.google.com/g/python_inside_maya/c/Y6r8o9zpWfU
#  also https://discourse.techart.online/t/pyqt-collapsible-groupbox-or-layout/1108/2

class MonoCalibrationGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Monocular Calibration')

        self.image_source = None
        self.calibration_pattern = None
        self.preprocessor = None

        # self.createActions()
        # self.createMenuBar()
        # self.createToolbars()
        self._initLayout()
        self._createStatusBar()
        # toolbar_imgsrc = QToolBar('Image Source', self)
        # self.addToolBar(Qt.TopToolBarArea, toolbar_imgsrc)

        self.resize(QDesktopWidget().availableGeometry(self).size() * 0.85)
    
    # def createActions(self):
    #     pass

    # def createMenuBars(self):
    #     pass

    # def createToolbars(self):
    #     pass
    
    def _initLayout(self):
        layout_main = QVBoxLayout()
        splitter_main = QSplitter(Qt.Vertical)
        layout_main.addWidget(splitter_main)
        ########### 1st row
        splitter_row1 = QSplitter(Qt.Horizontal)
        splitter_main.addWidget(splitter_row1)
        #### Image selection & pattern specification
        layout_row1 = QGridLayout()
        groupbox_input = QGroupBox("Images && Pattern")
        # groupbox_input.setCheckable(True) # TODO test (could be used to implement a collapsible box)
        groupbox_input.setLayout(QVBoxLayout())
        # groupbox_input.setStyleSheet("border: 1px solid gray; border-color: #ff17365d;")
        splitter_row1.addWidget(groupbox_input)

        self.calib_input = CalibrationInputWidget()
        self.calib_input.imageFolderSelected.connect(self._folderSelected)
        self.calib_input.patternConfigurationChanged.connect(self._patternConfigChanged)
        self.calib_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        groupbox_input.layout().addWidget(self.calib_input)

        #### Preprocessing pipeline
        groupbox_preproc = QGroupBox("Preprocessing")
        groupbox_preproc.setLayout(QVBoxLayout())
        dummy = PreprocessingSelector() #TODO
        dummy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        groupbox_preproc.layout().addWidget(dummy)
        splitter_row1.addWidget(groupbox_preproc)
        
        ########### 2nd row (image gallery)
        self.groupbox_imgview = QGroupBox("Images")
        self.groupbox_imgview.setLayout(QVBoxLayout())
        self.groupbox_imgview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # layout_main.addWidget(self.groupbox_imgview)
        splitter_main.addWidget(self.groupbox_imgview)
        from .ui.widgets.gallery import PlaceholderWidget
        self.groupbox_imgview.layout().addWidget(PlaceholderWidget())
        self.groupbox_imgview.setEnabled(False)

        splitter_main.setSizes([1, 3]) #TODO add 3rd row (results)
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
    def _folderSelected(self, folder):
        _logger.info(f'Loading images from {folder}')
        try:
            src = ImageDirectorySource(folder)
            self.groupbox_imgview.setEnabled(True) #TODO reset, then populate with images

            import time
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage(f'Loading {src.num_images()} images from {folder}')
            for i in range(src.num_images()):
                prg = int((i+1) / src.num_images() * 100)
                self.progress_bar.setValue(prg)
                time.sleep(0.1)
            self.status_bar.showMessage(f'Loaded {src.num_images()} images from {folder}', 10000)
        except (DirectoryNotFoundError, NoImageDirectoryError) as e:
            _logger.error("Error while loading image files:", exc_info=e)
            self.status_bar.showMessage(f'Error while loading images from {folder} ({e.__class__.__name__})')
            displayError(f"Error while loading images.", informative_text=str(e), parent=self)
            self.calib_input.resetImageFolder()
            #TODO self reset
    
    @Slot()
    def _patternConfigChanged(self):
        print('TODO pattern config changed')


# theme/style: https://pythonbasics.org/pyqt-style/
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    # print(QtWidgets.QStyleFactory.keys())
    # app.setStyle('Windows')
    # app.setStyle(QStyleFactory.create("Windows"))
    gui = MonoCalibrationGui()
    gui.show()
    app.exec_()
