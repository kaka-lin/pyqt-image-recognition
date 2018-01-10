from PyQt5 import QtCore, QtGui, QtWidgets
from gui.ui_videobox import Ui_VideoBox
from app.thread import VideoThread

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        """ """

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())

        # QWidgwe.update(self): 
        # calling update() several times normally results in just one paintEvent() call.
        self.update() 

class VideoBox(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(VideoBox, self).__init__(parent)

        self.image_viewer = ImageViewer()
    
        self.__threads = None

        self.ui = Ui_VideoBox()
        self.ui.setupUi(self)
        self._setup_ui()

    def _setup_ui(self):
        """ """
        self.ui.gridLayout.addWidget(self.image_viewer, 1, 0, 1, 1)

    def start(self):
        self.__threads = []
        video_worker = VideoThread()
        thread = QtCore.QThread(self)
        self.__threads.append((thread, video_worker))
        video_worker.moveToThread(thread)

        video_worker.image_data.connect(self.image_viewer.setImage)

        thread.started.connect(video_worker.startVideo)
        thread.start()
