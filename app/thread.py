import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoThread(QtCore.QThread):
    """ This thread is capture video with opencv """
    image_data = QtCore.pyqtSignal(QtGui.QImage)
    done_sig = QtCore.pyqtSignal()

    def __init__(self, camera_port=0, parent=None):
        super(VideoThread, self).__init__(parent)

        self.camera_port = camera_port
        self.running = False

    @QtCore.pyqtSlot()
    def startVideo(self):
        self.camera = cv2.VideoCapture(self.camera_port)
        #self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read() # type: np.ndarray

            if ret:
                # OpenCV stores data in BGR format. 
                # Qt stores data in RGB format.
                qt_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # type: np.ndarray
                #qt_image_2 = cv2.resize(qt_image, (640, 480), interpolation=cv2.INTER_AREA)
                height, width = qt_image.shape[:2]
                bytesPerLine = 3 * width # qt_image.strides[0] = 3 * width

                # QImage.Format_RGB888: 圖使用 8-8-8 24位RGB格式
                image = QtGui.QImage(qt_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

                self.image_data.emit(image)
        
        self.camera.release()
        cv2.destroyAllWindows()
        self.done_sig.emit()
    
    def stopVideo(self):
        self.running = False
