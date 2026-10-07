"""One-play splash screen. Plays the intro GIF exactly once, then emits `done`."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMovie, QGuiApplication
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class Splash(QWidget):
    done = pyqtSignal()

    SAFETY_MS = 20000  # never get stuck on the splash

    def __init__(self, gif_path):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #000000;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self._finished = False
        self.movie = QMovie(gif_path)
        if not self.movie.isValid():
            # GIF missing or unreadable: skip straight to the main window
            QTimer.singleShot(0, self._finish)
            return

        self.label.setMovie(self.movie)
        self.movie.frameChanged.connect(self._on_frame)
        self.movie.finished.connect(self._finish)        # non-looping GIFs
        QTimer.singleShot(self.SAFETY_MS, self._finish)  # safety fallback

        # Size the window to the GIF and center it on screen
        self.movie.jumpToFrame(0)
        size = self.movie.currentImage().size()
        if size.isValid():
            self.setFixedSize(size)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)

        self.movie.start()

    def _on_frame(self, n):
        count = self.movie.frameCount()
        if count > 0 and n == count - 1:                 # last frame reached
            self.movie.frameChanged.disconnect(self._on_frame)
            QTimer.singleShot(max(self.movie.nextFrameDelay(), 0), self._finish)

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self.movie.stop()
        self.close()
        self.done.emit()
