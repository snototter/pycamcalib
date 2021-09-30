"""Starts the monocular calibration GUI."""

#TODO move stereo/mono to .ui
from PySide2 import QtWidgets
from PySide2.QtCore import Qt, Slot
from PySide2.QtWidgets import QApplication, QDesktopWidget, QGroupBox, QMainWindow, QProgressBar, QSizePolicy, QStatusBar, QToolBar, QVBoxLayout, QWidget
import sys

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

        # self.createActions()
        # self.createMenuBar()
        # self.createToolbars()
        self.initLayout()
        self.setStatusBar(QStatusBar())
        # toolbar_imgsrc = QToolBar('Image Source', self)
        # self.addToolBar(Qt.TopToolBarArea, toolbar_imgsrc)
        self.resize(QDesktopWidget().availableGeometry(self).size() * 0.85)
    
    @Slot(str)
    def folderSelected(self, folder):
        print(f'TODO folder {folder} has been selected')
        self.groupbox_imgview.setEnabled(True) #TODO reset, then populate with images

        #TODO remove
        progress_bar = QProgressBar()
        self.statusBar().addWidget(progress_bar)
        from .calibration import ImageDirectorySource
        src = ImageDirectorySource(folder) # TODO catch FileNotFoundError
        import time
        for i in range(src.num_images()):
            prg = int((i+1) / src.num_images() * 100)
            progress_bar.setValue(prg)
            print(f'TODO progress {prg}')
            time.sleep(0.1)


    
    def initLayout(self):
        layout = QVBoxLayout()
        groupbox_input = QGroupBox("Input")
        groupbox_input.setCheckable(True) # TODO test
        groupbox_input.setLayout(QVBoxLayout())
        groupbox_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(groupbox_input)
        from .ui.widgets import ImageSourceSelectorWidget
        selector = ImageSourceSelectorWidget()
        selector.folderSelected.connect(self.folderSelected)
        groupbox_input.layout().addWidget(selector)

        groupbox_preproc = QGroupBox("Preprocessing")
        groupbox_preproc.setLayout(QVBoxLayout())
        groupbox_preproc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(groupbox_preproc)
        
        self.groupbox_imgview = QGroupBox("Images")
        self.groupbox_imgview.setLayout(QVBoxLayout())
        self.groupbox_imgview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.groupbox_imgview)
        from .ui.widgets.gallery import PlaceholderWidget
        self.groupbox_imgview.layout().addWidget(PlaceholderWidget())
        self.groupbox_imgview.setEnabled(False)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # groupbox_progress = QtWidgets.QGroupBox("Progress")
    # groupbox_progress.setLayout(QtWidgets.QHBoxLayout())
    # groupbox_progress.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    # layout.addWidget(groupbox_progress)
    # #TODO custom status bar widget (text + progressbar, or alternate text/progressbar)
    # pbar = QtWidgets.QProgressBar()
    # groupbox_progress.layout().addWidget(pbar)
    # pbar.setValue(80)

# theme/style: https://pythonbasics.org/pyqt-style/
if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(QtWidgets.QStyleFactory.keys())
    # app.setStyle('Windows')
    gui = MonoCalibrationGui()
    gui.show()
    app.exec_()
