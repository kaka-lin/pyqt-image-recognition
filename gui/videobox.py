import os
import sys

import cv2
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gui.ui_videobox import Ui_VideoBox
from gui.imageviewer import ImageViewer
from threads.video_thread import VideoThread
from models.models import CNN, BNN
from utils import *

# PyQt5 -> PySide2
QtCore.pyqtSignal = QtCore.Signal
QtCore.pyqtSlot = QtCore.Slot

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
