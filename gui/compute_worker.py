"""Runs the engine in a background thread so the window never freezes."""

import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)      # RunResult
    failed = pyqtSignal(str)

    def __init__(self, settings, entities, device):
        super().__init__()
        self.settings = settings
        self.entities = entities
        self.device = device
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from core.engine import Engine
            eng = Engine(self.settings, self.device)

            def cb(frac, msg):
                self.progress.emit(frac, msg)
                return not self._stop

            self.finished.emit(eng.run(self.entities, cb))
        except ValueError as exc:          # settings problems: plain message
            self.failed.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class ComputeJob(QObject):
    """Start with ComputeJob(settings, entities, device).start(); stop with .stop()."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings, entities, device, parent=None):
        super().__init__(parent)
        self.thread = QThread()
        self.worker = _Worker(settings, entities, device)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._fail)

    def start(self):
        self.thread.start()

    def stop(self):
        self.worker.stop()

    def _done(self, res):
        self.thread.quit(); self.thread.wait()
        self.finished.emit(res)

    def _fail(self, msg):
        self.thread.quit(); self.thread.wait()
        self.failed.emit(msg)
