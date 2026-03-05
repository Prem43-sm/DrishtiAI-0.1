from PySide6.QtWidgets import (
    QMessageBox, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap

import os
import subprocess
from datetime import datetime

from features.multi_camera.multi_camera_manager import MultiCameraManager


# ================= REPORT WORKER =================
class ReportWorker(QThread):
    finished = Signal(bool, str)

    def run(self):
        try:
            result = subprocess.run(
                ["python", "monthly_report.py"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.finished.emit(True, "Monthly report generated ✅")
            else:
                self.finished.emit(False, result.stderr)

        except Exception as e:
            self.finished.emit(False, str(e))


# ================= DASHBOARD =================
class DashboardPage(QWidget):

    def __init__(self):
        super().__init__()

        self.cam_manager = MultiCameraManager()
        self.worker = None
        self.report_worker = None
        self.current_frame = None
        self._last_popup_msg = ""
        self._last_popup_ts = 0.0
        self._popup_box = None

        main_layout = QVBoxLayout()

        title = QLabel("Dashboard")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        # 🎥 CAMERA DISPLAY
        self.camera_frame = QLabel()
        self.camera_frame.setMinimumSize(320, 180)
        self.camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_frame.setAlignment(Qt.AlignCenter)
        self.camera_frame.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.camera_frame, 1)

        # 🧠 INFO LABELS
        info_layout = QHBoxLayout()

        self.name_label = QLabel("Name: ---")
        self.emotion_label = QLabel("Emotion: ---")
        self.fps_label = QLabel("FPS: 0")
        self.model_label = QLabel("Model: Not Loaded")

        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.emotion_label)
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.model_label)

        main_layout.addLayout(info_layout)

        # 🎮 BUTTONS
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.snapshot_btn = QPushButton("Snapshot")
        self.reload_model_btn = QPushButton("Reload Model")
        self.report_btn = QPushButton("Generate Monthly Report")

        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.snapshot_btn)
        btn_layout.addWidget(self.reload_model_btn)
        btn_layout.addWidget(self.report_btn)

        main_layout.addLayout(btn_layout)

        # 📊 STATS
        stats_layout = QHBoxLayout()

        self.attendance_count = QLabel("Today Attendance: 0")
        self.known_faces_count = QLabel("Known Faces: 0")

        stats_layout.addWidget(self.attendance_count)
        stats_layout.addWidget(self.known_faces_count)

        main_layout.addLayout(stats_layout)

        self.setLayout(main_layout)

        # 🔘 BUTTON CONNECTIONS
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.report_btn.clicked.connect(self.generate_report)

    # ================= START CAMERA =================
    def start_camera(self):

        self.cam_manager.start_all()

        # take first camera worker for display
        self.worker = self.cam_manager.workers[0]
        self.worker.frame_ready.connect(self.update_ui)

        self.model_label.setText("Model: Loaded ✅")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        if os.path.exists("known_faces"):
            count = len(os.listdir("known_faces"))
            self.known_faces_count.setText(f"Known Faces: {count}")

    # ================= STOP CAMERA =================
    def stop_camera(self):

        self.cam_manager.stop_all()
        self.worker = None

        self.camera_frame.clear()

        self.name_label.setText("Name: ---")
        self.emotion_label.setText("Emotion: ---")
        self.fps_label.setText("FPS: 0")
        self.model_label.setText("Model: Not Loaded")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._close_attendance_popup()

    # ================= UPDATE UI =================
    def update_ui(self, image, name, emotion, fps, count, popup_msg):

        self.current_frame = image
        self._render_current_frame()

        self.name_label.setText(f"Name: {name}")
        self.emotion_label.setText(f"Emotion: {emotion}")
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.attendance_count.setText(f"Today Attendance: {count}")

        if popup_msg:
            self.show_attendance_popup(popup_msg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current_frame()

    def _render_current_frame(self):
        if self.current_frame is None:
            return
        self.camera_frame.setPixmap(
            QPixmap.fromImage(self.current_frame).scaled(
                self.camera_frame.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )

    def show_attendance_popup(self, text):
        now = datetime.now().timestamp()
        # Avoid duplicate popup spam for same message in a short window.
        if text == self._last_popup_msg and (now - self._last_popup_ts) < 2:
            return

        self._last_popup_msg = text
        self._last_popup_ts = now

        self._close_attendance_popup()

        self._popup_box = QMessageBox(self)
        self._popup_box.setWindowTitle("Attendance")
        self._popup_box.setText(text)
        self._popup_box.setIcon(QMessageBox.Information)
        self._popup_box.setStandardButtons(QMessageBox.NoButton)
        self._popup_box.setModal(False)
        self._popup_box.setAttribute(Qt.WA_DeleteOnClose, True)
        self._popup_box.show()

        QTimer.singleShot(5000, self._close_attendance_popup)

    def _close_attendance_popup(self):
        if self._popup_box is None:
            return
        try:
            self._popup_box.accept()
            self._popup_box.deleteLater()
        finally:
            self._popup_box = None

    # ================= SNAPSHOT =================
    def take_snapshot(self):

        if self.current_frame is None:
            return

        os.makedirs("snapshots", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"snapshots/snapshot_{timestamp}.png"

        self.current_frame.save(file_path)

    # ================= GENERATE REPORT =================
    def generate_report(self):

        self.report_btn.setEnabled(False)
        self.report_btn.setText("Generating...")

        self.report_worker = ReportWorker()
        self.report_worker.finished.connect(self.report_done)
        self.report_worker.start()

    def report_done(self, success, message):

        self.report_btn.setEnabled(True)
        self.report_btn.setText("Generate Monthly Report")

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Error", message)
