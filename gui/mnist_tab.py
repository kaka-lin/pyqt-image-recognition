from PyQt5 import QtCore, QtGui, QtWidgets

class MnistTab(QtWidgets.QVBoxLayout):
    def __init__(self, parent=None):
        super(MnistTab, self).__init__(parent)
      
        self._setup_ui()
    
    def _setup_ui(self):
        self.retranslateUi()
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
