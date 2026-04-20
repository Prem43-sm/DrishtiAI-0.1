import math
from datetime import datetime

import cv2
from PySide6.QtCore import QThread, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.camera_backend import open_camera_capture, scan_camera_ids


class CameraPreviewWorker(QThread):
    frame_ready = Signal(int, QImage)
    camera_error = Signal(int, str)

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.running = True

    def run(self):
        cap, _, _ = open_camera_capture(self.camera_id)
        if cap is None:
            self.camera_error.emit(self.camera_id, "Camera open failed")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(self.camera_id, img)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class ClickableVideoLabel(QLabel):
    clicked = Signal(int)

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)


class FullScreenCameraWindow(QWidget):
    def __init__(self, camera_name):
        super().__init__()
        self.setWindowTitle(f"{camera_name} - Live View")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        self.video = QLabel("Loading...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:black; color:#9aa0a6;")
        layout.addWidget(self.video)

    def update_image(self, image):
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


class MultiCameraViewPage(QWidget):
    def __init__(self):
        super().__init__()

        self.max_scan = 8
        self.camera_ids = []
        self.workers = {}
        self.camera_labels = {}
        self.camera_names = {}
        self.latest_frames = {}

        self.fullscreen_window = None
        self.fullscreen_camera_id = None
        self._scanned_once = False

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QHBoxLayout()

        title = QLabel("Multi Camera View")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        header.addWidget(title)
        header.addStretch()

        self.live_datetime_label = QLabel()
        self.live_datetime_label.setStyleSheet(
            "color: yellow; font-size:14px; font-weight:bold;"
        )
        header.addWidget(self.live_datetime_label)
        root.addLayout(header)

        controls = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.start_btn = QPushButton("Start Live View")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        root.addLayout(controls)

        self.status_label = QLabel("Status: Idle")
        root.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.grid_host)
        root.addWidget(self.scroll)

        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.start_btn.clicked.connect(self.start_preview)
        self.stop_btn.clicked.connect(self.stop_preview)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_live_datetime)
        self.clock_timer.start(1000)
        self._update_live_datetime()

    def _update_live_datetime(self):
        now = datetime.now().strftime("LIVE  %d-%m-%Y  %H:%M:%S")
        self.live_datetime_label.setText(now)

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.camera_labels.clear()

    def _scan_connected_cameras(self):
        return scan_camera_ids(self.max_scan)

    def refresh_cameras(self):
        self.stop_preview()
        self.camera_ids = self._scan_connected_cameras()
        self.camera_names = {cid: f"cam-{i+1:02d}" for i, cid in enumerate(self.camera_ids)}
        self._build_camera_grid()
        self._scanned_once = True
        if self.camera_ids:
            self.status_label.setText(f"Status: Found {len(self.camera_ids)} camera(s)")
        else:
            self.status_label.setText("Status: No camera detected")

    def _build_camera_grid(self):
        self._clear_grid()
        if not self.camera_ids:
            note = QLabel("No camera connected")
            note.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(note, 0, 0)
            return

        count = len(self.camera_ids)
        if count == 1:
            cols = 1
        elif count <= 4:
            cols = 2
        else:
            cols = max(3, math.ceil(math.sqrt(count)))

        rows = math.ceil(count / cols)

        for i, camera_id in enumerate(self.camera_ids):
            row = i // cols
            col = i % cols

            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            v = QVBoxLayout(card)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)

            video = ClickableVideoLabel(camera_id)
            video.setMinimumSize(320, 220)
            video.setAlignment(Qt.AlignCenter)
            video.setText("Click to open full view")
            video.setStyleSheet("background:black; color:#9aa0a6;")
            video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            video.clicked.connect(self.open_fullscreen_camera)
            v.addWidget(video)

            self.camera_labels[camera_id] = video
            self.grid.addWidget(card, row, col)

        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
        for r in range(rows):
            self.grid.setRowStretch(r, 1)

    def start_preview(self):
        self.stop_preview()
        if not self.camera_ids:
            self.status_label.setText("Status: No camera to start")
            return

        for camera_id in self.camera_ids:
            w = CameraPreviewWorker(camera_id)
            w.frame_ready.connect(self.update_frame)
            w.camera_error.connect(self.on_camera_error)
            self.workers[camera_id] = w
            w.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Live view running")

    def stop_preview(self):
        for w in list(self.workers.values()):
            w.stop()
        self.workers.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_camera_error(self, camera_id, msg):
        self.status_label.setText(f"Status: cam-{camera_id} error: {msg}")

    def update_frame(self, camera_id, image):
        frame = image.copy()
        camera_name = self.camera_names.get(camera_id, f"cam-{camera_id:02d}")

        painter = QPainter(frame)
        painter.setRenderHint(QPainter.Antialiasing)
        f = QFont()
        # dynamic text size so label remains visible on resize/smaller previews
        point_size = max(11, min(18, frame.height() // 28))
        f.setPointSize(point_size)
        f.setBold(True)
        painter.setFont(f)

        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(camera_name)
        text_h = metrics.height()

        pad_x = 10
        pad_y = 6
        box_x = 8
        box_y = 8
        box_w = text_w + (pad_x * 2)
        box_h = text_h + (pad_y * 2)

        # solid dark strip for guaranteed readability
        painter.fillRect(box_x, box_y, box_w, box_h, QColor(0, 0, 0, 190))
        painter.setPen(QColor(255, 255, 0))
        text_x = box_x + pad_x
        text_y = box_y + pad_y + metrics.ascent()
        painter.drawText(text_x, text_y, camera_name)
        painter.end()

        self.latest_frames[camera_id] = frame

        label = self.camera_labels.get(camera_id)
        if label:
            label.setPixmap(
                QPixmap.fromImage(frame).scaled(
                    label.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
            )

        if self.fullscreen_window and self.fullscreen_camera_id == camera_id:
            self.fullscreen_window.update_image(frame)

    def open_fullscreen_camera(self, camera_id):
        camera_name = self.camera_names.get(camera_id, f"cam-{camera_id}")
        self.fullscreen_camera_id = camera_id

        if self.fullscreen_window:
            self.fullscreen_window.close()

        self.fullscreen_window = FullScreenCameraWindow(camera_name)
        self.fullscreen_window.showMaximized()

        frame = self.latest_frames.get(camera_id)
        if frame is not None:
            self.fullscreen_window.update_image(frame)

    def closeEvent(self, event):
        self.stop_preview()
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._scanned_once:
            self.refresh_cameras()
