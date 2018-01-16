import cv2
import numpy as np
from keras.models import Sequential, load_model
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoThread(QtCore.QThread):
    """ This thread is capture video with opencv """
    image_data = QtCore.pyqtSignal(np.ndarray)
    done_sig = QtCore.pyqtSignal()
    predict_sig = QtCore.pyqtSignal(['QString'])

    def __init__(self, model, camera_port=0, parent=None):
        super(VideoThread, self).__init__(parent)

        self.model = model
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
        
        self.camera.release()
        cv2.destroyAllWindows()
        self.done_sig.emit()
    
    def stopVideo(self):
        self.running = False
