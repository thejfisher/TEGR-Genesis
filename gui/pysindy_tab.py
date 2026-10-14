"""PySINDy Analysis Tab."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
    QSplitter, QDoubleSpinBox, QLabel, QApplication, QScrollArea, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class PySindyTab(QWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        
        # Top bar
        top = QHBoxLayout()
        self.btn_run = QPushButton("Run PySINDy Analysis")
        self.btn_run.setObjectName("runSindyButton")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_analysis)
        
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.001, 1.0)
        self.thresh_spin.setDecimals(3)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.05)
        self.thresh_spin.setToolTip("STLSQ Threshold (sparsification)")
        
        top.addWidget(self.btn_run)
        top.addWidget(QLabel("Threshold:"))
        top.addWidget(self.thresh_spin)
        
        self.chk_gravity = QCheckBox("Gravity (1/r²)")
        self.chk_gravity.setChecked(True)
        self.chk_gamma = QCheckBox("Relativity (γ, m0/γ)")
        self.chk_phase = QCheckBox("Phase/Solitons (θ, sin(θ), cos(θ))")
        
        top.addWidget(self.chk_gravity)
        top.addWidget(self.chk_gamma)
        top.addWidget(self.chk_phase)
        top.addStretch()
        lay.addLayout(top)
        
        # Splitter for text and graph
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.text_out = QTextEdit()
        self.text_out.setReadOnly(True)
        self.text_out.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.text_out.setPlaceholderText("PySINDy analysis results will appear here...")
        self.splitter.addWidget(self.text_out)
        
        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_container = QWidget()
        self.canvas_lay = QVBoxLayout(self.canvas_container)
        self.canvas_scroll.setWidget(self.canvas_container)
        self.splitter.addWidget(self.canvas_scroll)
        
        self.splitter.setSizes([450, 550])
        lay.addWidget(self.splitter)

    def set_result(self, res):
        self.result = res
        self.btn_run.setEnabled(res is not None)
        if res is None:
            self.text_out.clear()
            self._clear_canvas()

    def _clear_canvas(self):
        while self.canvas_lay.count():
            child = self.canvas_lay.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _run_analysis(self):
        if self.result is None:
            return
            
        self.btn_run.setEnabled(False)
        self.text_out.setPlainText("Running PySINDy... Please wait.")
        self._clear_canvas()
        QApplication.processEvents()
        
        try:
            from core.exporter import sindy_fit, fig_sindy
            threshold = self.thresh_spin.value()
            opts = {
                "gravity": self.chk_gravity.isChecked(),
                "gamma": self.chk_gamma.isChecked(),
                "phase": self.chk_phase.isChecked(),
            }
            text, fits = sindy_fit(self.result, threshold=threshold, options=opts)
            
            self.text_out.setPlainText(text)
            
            fig = fig_sindy(fits)
            if fig is not None:
                canvas = FigureCanvasQTAgg(fig)
                self.canvas_lay.addWidget(canvas)
                
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.text_out.setPlainText(f"PySINDy Error:\n\n{exc}")
        finally:
            self.btn_run.setEnabled(True)
