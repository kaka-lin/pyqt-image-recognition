from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSizePolicy

class Ui_VideoBox(object):
    def setupUi(self, videoBox):
        videoBox.setObjectName("videoBox")

        self.gridLayout = QtWidgets.QGridLayout(videoBox)
        self.gridLayout.setHorizontalSpacing(12)
        self.gridLayout.setVerticalSpacing(12)

        self.run_button = QtWidgets.QPushButton(videoBox)
        self.run_button.setObjectName("run_button")

        self.gridLayout.addWidget(self.run_button, 0, 0, 1, 1)
        
        self.run_button.clicked.connect(videoBox.start)
        self.retranslateUi(videoBox)

    def retranslateUi(self, videoBox):
        _translate = QtCore.QCoreApplication.translate
        #videoBox.setTitle(_translate("VideoBox", "Image Processing"))
        self.run_button.setText(_translate("VideoBox", "Start"))
        
    