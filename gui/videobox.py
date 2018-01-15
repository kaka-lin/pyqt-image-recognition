import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from gui.ui_videobox import Ui_VideoBox
from app.thread import VideoThread

from keras.models import Sequential, load_model

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
        self.image = QtGui.QImage()
        
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
        # OpenCV stores data in BGR format. 
        # Qt stores data in RGB format.
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

        path = sys.path[0]
        self.model = load_model(path + '/cnn_demo_model.h5')
        # 初始化加載模型後，需随便生成一个向量讓model先執行一次predict
        # 之後使用才不會出現 ValueError
        self.model.predict(np.zeros((1, 28, 28, 1)))

        self.ui = Ui_VideoBox()
        self.ui.setupUi(self)
        self._setup_ui()

    def _setup_ui(self):
        """ """
        self.ui.gridLayout.addWidget(self.image_viewer, 1, 0, 1, 2)

    def start(self):
        self.__threads = []
        thread = QtCore.QThread(self)
        video_worker = VideoThread(self.model)
        self.__threads.append((thread, video_worker))
        video_worker.moveToThread(thread)

        video_worker.image_data.connect(self.image_viewer.setImage)
        video_worker.done_sig.connect(self.on_video_done)
        video_worker.predict_sig.connect(self.on_predict_update)
        
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

        self.image_viewer.update()
        print('Video Thread Finished!')

    @QtCore.pyqtSlot('QString')
    def on_predict_update(self, predict):
        self.ui.predict_label.setText('Predict Label: {0}'.format(predict))
