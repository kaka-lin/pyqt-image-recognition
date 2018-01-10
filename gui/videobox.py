from PyQt5 import QtCore, QtGui, QtWidgets
from gui.ui_videobox import Ui_VideoBox
from app.thread import VideoThread

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

class VideoBox(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super(VideoBox, self).__init__(parent)

        self.image_viewer = ImageViewer()
        self.video_worker = VideoThread()
    
        self.__threads = None

        self.ui = Ui_VideoBox()
        self.ui.setupUi(self)
        self._setup_ui()

    def _setup_ui(self):
        """ """
        self.ui.gridLayout.addWidget(self.image_viewer, 1, 0, 1, 2)

    def start(self):
        self.__threads = []
        thread = QtCore.QThread(self)
        self.__threads.append((thread, self.video_worker))
        self.video_worker.moveToThread(thread)

        self.video_worker.image_data.connect(self.image_viewer.setImage)
        self.video_worker.done_sig.connect(self.on_video_done)
        
        thread.started.connect(self.video_worker.startVideo)
        thread.start()
    
    def stop(self):
        self.video_worker.stopVideo()
    
    @QtCore.pyqtSlot()
    def on_video_done(self):
        for thread, worker in self.__threads:
            thread.quit()
            thread.wait()

        self.image_viewer.update()
        print('Video Thread Finished!')
