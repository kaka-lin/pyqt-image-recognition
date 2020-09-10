import cv2
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

# PyQt5 -> PySide2
QtCore.pyqtSignal = QtCore.Signal
QtCore.pyqtSlot = QtCore.Slot

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
