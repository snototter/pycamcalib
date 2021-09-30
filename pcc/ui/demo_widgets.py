"""Lists all custom controls/widgets"""
import sys
from PySide2 import QtCore, QtWidgets
from .widgets import ImageSourceSelectorWidget


@QtCore.Slot(str)
def slot_str(s):
    print(f'Signal(str) received: "{s}"')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(main)

    groupbox_input = QtWidgets.QGroupBox("Input")
    groupbox_input.setLayout(QtWidgets.QVBoxLayout())
    groupbox_input.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    layout.addWidget(groupbox_input)
    selector = ImageSourceSelectorWidget()
    selector.folderSelected.connect(slot_str)
    groupbox_input.layout().addWidget(selector)

    groupbox_preproc = QtWidgets.QGroupBox("Preprocessing")
    groupbox_preproc.setLayout(QtWidgets.QVBoxLayout())
    groupbox_preproc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    layout.addWidget(groupbox_preproc)
    
    groupbox_imgview = QtWidgets.QGroupBox("Images")
    groupbox_imgview.setLayout(QtWidgets.QVBoxLayout())
    groupbox_imgview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    layout.addWidget(groupbox_imgview)
    from .widgets.gallery import PlaceholderWidget
    groupbox_imgview.layout().addWidget(PlaceholderWidget())

    groupbox_progress = QtWidgets.QGroupBox("Progress")
    groupbox_progress.setLayout(QtWidgets.QHBoxLayout())
    groupbox_progress.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    layout.addWidget(groupbox_progress)
    #TODO custom status bar widget (text + progressbar, or alternate text/progressbar)
    pbar = QtWidgets.QProgressBar()
    groupbox_progress.layout().addWidget(pbar)
    pbar.setValue(80)

    main.show()
    sys.exit(app.exec_())