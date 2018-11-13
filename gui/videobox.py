import os
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gui.ui_videobox import Ui_VideoBox
from app.thread import VideoThread
from models.models import CNN, BNN
from utils import *

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()

        self._setup_ui()

    def _setup_ui(self):
        """ """

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        #self.image = QtGui.QImage()
        
    @QtCore.pyqtSlot(np.ndarray)
    def setImage(self, frame):
        image = self.get_qt_image(frame)
        self.image = image
        
        if image.size() != self.size():
            self.setFixedSize(image.size())
    
        # QWidgwe.update(self): 
        # calling update() several times normally results in just one paintEvent() call.
        self.update()
    
    def get_qt_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 150), interpolation=cv2.INTER_AREA)

        height, width = image.shape[:2]
        bytesPerLine = 3 * width # image.strides[0] = 3 * width

        qt_image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        return qt_image

class VideoBox(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super(VideoBox, self).__init__(parent)

        self.image_viewer = ImageViewer()
        self.__threads = None
        self.model = None
        
        self.ui = Ui_VideoBox()
        self.ui.setupUi(self)
        self._setup_ui()

    def _setup_ui(self):
        """ """
        self.ui.gridLayout.addWidget(self.image_viewer, 1, 0, 1, 2)

    def start(self):
        model_name = self.choice_model()

        self.__threads = []
        thread = QtCore.QThread(self)
        video_worker = VideoThread(self.model, model_name)
        self.__threads.append((thread, video_worker))
        video_worker.moveToThread(thread)

        video_worker.image_data.connect(self.image_viewer.setImage)
        video_worker.done_sig.connect(self.on_video_done)
        video_worker.predict_sig.connect(self.update_lable)
        
        thread.started.connect(video_worker.startVideo)
        thread.start()
    
    def stop(self):
        for thread, worker in self.__threads:
            worker.stopVideo()
    
    @QtCore.pyqtSlot()
    def on_video_done(self):
        for thread, worker in self.__threads:
            thread.quit()
            thread.wait()

        self.image_viewer.image = QtGui.QImage()
        self.image_viewer.update()
        print('Video Thread Finished!')
    
    @QtCore.pyqtSlot('QString')
    def update_lable(self, label):
        print('Predict Label: {0}'.format(label))
        self.ui.predict_label.setText('Predict Label: {0}'.format(label))
    
    def choice_model(self):
        model_name = self.ui.combobox.currentText()
        path = os.path.join(sys.path[0], 'models/pre_trains/')

        if model_name == 'BNN':
            # Load model    
            self.model = BNN()
            self.loss = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters())
            
            path = os.path.join(sys.path[0], 'models/pre_trains/')    
            checkpoint = torch.load(path + 'binary_model_checkpoint.tar')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
        else:
            self.model = load_model(path + 'cnn_demo_model.h5')
            # 初始化加載模型後，需随便生成一个向量讓model先執行一次predict
            # 之後使用才不會出現 ValueError
            self.model.predict(np.zeros((1, 28, 28, 1)))
        
        return model_name
