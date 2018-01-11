import cv2
import numpy as np
from keras.models import Sequential, load_model
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoThread(QtCore.QThread):
    """ This thread is capture video with opencv """
    image_data = QtCore.pyqtSignal(np.ndarray)
    done_sig = QtCore.pyqtSignal()

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
                self.test(frame)
                self.image_data.emit(frame)
        
        self.camera.release()
        cv2.destroyAllWindows()
        self.done_sig.emit()
    
    def stopVideo(self):
        self.running = False
    
    def test(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        image = cv2.resize(thresh1, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite('x_test.png', image)
        image = image.reshape(1, 28*28)
        image = image.astype('float32')
        image = image / 255

        print('=========================')
        x_test = image.reshape(1, 28, 28, 1)

        scores = self.model.predict(x_test)
        top_label_ix = np.argmax(scores)
        print("Predict Label: {0}".format(top_label_ix))
        print('=========================')

       