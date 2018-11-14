import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision.transforms as transforms

class VideoThread(QtCore.QThread):
    """ This thread is capture video with opencv """
    image_data = QtCore.pyqtSignal(np.ndarray)
    done_sig = QtCore.pyqtSignal()
    predict_sig = QtCore.pyqtSignal('QString')

    def __init__(self, model, model_name, camera_port=0, parent=None):
        super(VideoThread, self).__init__(parent)

        self.model = model
        self.model_name = model_name
        self.camera_port = camera_port
        self.running = False

    @QtCore.pyqtSlot()
    def startVideo(self):
        self.camera = cv2.VideoCapture(self.camera_port)
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read()

            if ret:
                self.image_data.emit(frame)
                self.start_predict(frame)
        
        self.camera.release()
        cv2.destroyAllWindows()
        self.done_sig.emit()
    
    def stopVideo(self):
        self.running = False
    
    def start_predict(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        image = cv2.resize(thresh1, (28, 28), interpolation=cv2.INTER_AREA)

        cv2.imwrite('x_test.png', image)

        image = image.reshape(1, 28*28)
        image = image.astype('float32')
        image = image / 255

        if self.model_name == 'BNN':
            # Normalize
            image = ((image - 0.1307) / 0.3081)
            x_test = image.reshape(1, 1, 28, 28)
            x_test = torch.from_numpy(x_test)

            self.model.eval()
            output = self.model(x_test)
            predict = str((output.max(1, keepdim=True)[1]).item())
        else:
            x_test = image.reshape(1, 28, 28, 1)
            scores = self.model.predict(x_test)
            predict = str(np.argmax(scores))

        self.predict_sig.emit(predict)

        return predict
