import sys
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader

from pathlib import Path
UI_PATH = Path(__file__).resolve().parent / "rs2_concept.ui"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load .ui
        loader = QUiLoader()
        ui_file = QtCore.QFile(UI_PATH)
        if not ui_file.open(QtCore.QFile.ReadOnly):
            raise RuntimeError(f"Failed to open UI file: {UI_PATH}")
        self.ui = loader.load(ui_file)
        ui_file.close()
        if self.ui is None:
            raise RuntimeError(f"Failed to load UI file: {UI_PATH}")

        # The .ui root is a QMainWindow. Reuse its central widget.
        self.setCentralWidget(self.ui.centralWidget())
        self.setWindowTitle(self.ui.windowTitle())

        # --- CAMERA TAB SETUP ---
        # Your QTabWidget is named "mainTab" in the .ui.
        self.tabs: QtWidgets.QTabWidget = self.findChild(QtWidgets.QTabWidget, "mainTab")
        if self.tabs is None:
            raise RuntimeError("Could not find QTabWidget named 'mainTab' in UI.")

        # Assume tab_2 is your camera tab (2nd tab index = 1)
        self.camera_tab: QtWidgets.QWidget = self.tabs.widget(1)

        # Add a layout + QLabel to fill the camera tab
        layout = self.camera_tab.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.camera_tab)
            layout.setContentsMargins(6, 6, 6, 6)

        self.lblCamera = QtWidgets.QLabel("Camera feed will appear here", self.camera_tab)
        self.lblCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCamera.setMinimumSize(320, 240)
        self.lblCamera.setScaledContents(False)  # we will scale pixmap ourselves
        layout.addWidget(self.lblCamera)

        # Open laptop camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.lblCamera.setText("❌ Could not open laptop camera (index 0).")
            self.cap = None
            return

        # Timer to update frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)  # ~30 FPS

    def _update_frame(self):
        if self.cap is None:
            return

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.lblCamera.setText("⚠️ Camera frame read failed.")
            return

        # Convert BGR (OpenCV) -> RGB (Qt expects RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Scale to fit label while keeping aspect ratio
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.lblCamera.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.lblCamera.setPixmap(pixmap)

    def closeEvent(self, event: QtGui.QCloseEvent):
        # Clean shutdown
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
