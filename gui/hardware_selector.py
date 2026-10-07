"""Hardware picker: Auto / CUDA / Mac (MPS) / CPU, with automatic fallback to CPU.

Detection lives in core/hardware.py (no Qt) so the engine uses the same rules.
PyTorch is imported lazily, so the splash screen is never slowed down.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox

from core.hardware import available_devices, resolve_device

CHOICES = ["Auto", "CUDA (NVIDIA)", "Mac (MPS)", "CPU"]
_KEYS = {"Auto": "auto", "CUDA (NVIDIA)": "cuda", "Mac (MPS)": "mps", "CPU": "cpu"}



class HardwareSelector(QWidget):
    """Small combo box. Emits `device_changed(device_name, note)`."""

    device_changed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._info = None
        self.device = "cpu"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Hardware:")
        self.combo = QComboBox()
        self.combo.setObjectName("hardwareCombo")
        self.combo.addItems(CHOICES)
        self.combo.setToolTip("Where the physics runs. Falls back to CPU if needed.")
        self.status = QLabel("checking…")
        self.status.setObjectName("hardwareStatus")
        self.status.setMinimumWidth(190)
        layout.addWidget(label)
        layout.addWidget(self.combo)
        layout.addWidget(self.status)

        self.combo.currentTextChanged.connect(self._apply)

    def detect(self):
        """Look for GPUs (imports PyTorch). Call after the main window is visible."""
        self._info = available_devices()
        self._apply(self.combo.currentText())

    def set_choice(self, key):
        for text, k in _KEYS.items():
            if k == key:
                self.combo.setCurrentText(text)
                return

    def choice_key(self):
        return _KEYS[self.combo.currentText()]

    def _apply(self, text):
        if self._info is None:
            return
        self.device, note = resolve_device(_KEYS[text], self._info)
        shown = {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}[self.device]
        if self.device == "cuda" and self._info.get("cuda_name"):
            short = self._info["cuda_name"].replace("NVIDIA ", "").replace("GeForce ", "")
            shown = f"CUDA · {short}"
            note = (note + " — " if note else "") + self._info["cuda_name"]
        self.status.setText(f"● {shown}")
        self.status.setToolTip(note)
        self.device_changed.emit(self.device, note)
