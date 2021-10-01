from PySide2.QtWidgets import QListView, QWidget

class PreprocessingSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.list_view = QListView()
