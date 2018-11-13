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
        self.stop_button = QtWidgets.QPushButton(videoBox)
        self.stop_button.setObjectName("stop_button")
        self.predict_label = QtWidgets.QLabel(videoBox)
        self.predict_label.setObjectName("predict_label")
        self.combobox = QtWidgets.QComboBox(videoBox)

        self.gridLayout.addWidget(self.combobox, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.run_button, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.stop_button, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.predict_label, 2, 0, 1, 2)
        
        self.run_button.clicked.connect(videoBox.start)
        self.stop_button.clicked.connect(videoBox.stop)

        self.retranslateUi(videoBox)

    def retranslateUi(self, videoBox):
        _translate = QtCore.QCoreApplication.translate
        videoBox.setTitle(_translate("VideoBox", "Video Stream"))
        self.run_button.setText(_translate("VideoBox", "Start"))
        self.stop_button.setText(_translate("VideoBox", "Stop"))
        self.predict_label.setText(_translate("VideoBox", "Predict Label: None"))

        self.combobox.addItems(["CNN", "BNN"])
        
    