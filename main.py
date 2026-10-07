"""TEGR Genesis — entry point. Shows the one-play splash, then the main window."""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from gui.splash import Splash
from gui.main_window import MainWindow

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SPLASH_GIF = ASSETS / "TEGR_retro_logo_intro.gif"
THEME_QSS = ASSETS / "genesis_theme.qss"


def load_theme(app):
    try:
        app.setStyleSheet(THEME_QSS.read_text(encoding="utf-8"))
    except OSError:
        pass  # run unstyled rather than crash


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TEGR Genesis")
    load_theme(app)

    window = MainWindow()
    splash = Splash(str(SPLASH_GIF))

    def open_main():
        window.show()
        window.raise_()
        window.activateWindow()

    splash.done.connect(open_main)
    splash.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
