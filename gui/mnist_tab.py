from PyQt5 import QtCore, QtGui, QtWidgets
from gui.videobox import VideoBox

class MnistTab(QtWidgets.QVBoxLayout):
    def __init__(self, parent=None):
        super(MnistTab, self).__init__(parent)

        self.videobox = VideoBox()
        self.addWidget(self.videobox)

        self._setup_ui()
    
    def _setup_ui(self):
        self.retranslateUi()
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
