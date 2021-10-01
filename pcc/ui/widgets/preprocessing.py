from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
from ...processing import Preprocessor, AVAILABLE_PREPROCESSOR_OPERATIONS

class AddOperationItem(QWidget):
    addOperation = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        #TODO add spacer (for operation number)
        #TODO add combobox (op selection)
        btn = QPushButton("Add")
        btn.clicked.connect(self._triggerAdd)
        layout.addWidget(btn)
        self.setLayout(layout)
    
    @Slot()
    def _triggerAdd(self):
        self.addOperation.emit('op name')


class PreprocessingSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preprocessor = Preprocessor()

        self._initLayout()
    
    def _initLayout(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        for opcls in AVAILABLE_PREPROCESSOR_OPERATIONS:
            print(f'TODO add op to list: {opcls.name} "{opcls.description}"')
        #TODO add grayscale by default!
        # Add item
        item = QListWidgetItem(self.list_widget)
        self.list_widget.addItem(item)
        # Instanciate a custom widget 
        item_widget = AddOperationItem()
        item.setSizeHint(item_widget.minimumSizeHint())
        self.list_widget.setItemWidget(item, item_widget)

        self.setLayout(layout)
