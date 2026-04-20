import os
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QProgressBar,
    QFrame,
)

from core.project_paths import NOISE_ALERTS_DIR, ensure_runtime_layout
try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
except Exception:
    QAudioFormat = None
    QAudioSource = None
    QMediaDevices = None

from gui.emotion_model_runtime import get_misbehavior_alert_emotions
from features.multi_camera.multi_camera_manager import MultiCameraManager
from features.tracking.live_tracker import get_all_locations


class BehaviorPage(QWidget):
    def __init__(self):
        super().__init__()
        ensure_runtime_layout()

        self.cam_manager = MultiCameraManager()
        self.worker = None
        self.current_frame = None

        self.audio_source = None
        self.audio_device = None
        self.noise_level = 0

        self.noise_threshold = 55
        self.last_snapshot_time = 0.0
        self.snapshot_cooldown_sec = 8
        self.last_alert_by_person = {}
        self.alert_cooldown_sec = 6
        self.alert_emotions = get_misbehavior_alert_emotions()

        self._build_ui()

        self.people_timer = QTimer(self)
        self.people_timer.timeout.connect(self.refresh_people_list)
        self.people_timer.start(1000)
        self.snapshot_timer = QTimer(self)
        self.snapshot_timer.timeout.connect(self.refresh_snapshot_list)
        self.snapshot_timer.start(2500)

        self.refresh_people_list()
        self.refresh_snapshot_list()

    def _build_ui(self):
        main = QVBoxLayout()
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)

        title = QLabel("Noise and Misbehavior")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main.addWidget(title)

        top_card = QFrame()
        top_card.setObjectName("Card")
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(12, 12, 12, 12)
        top_layout.setSpacing(10)

        self.camera_label = QLabel("Camera feed not started")
        self.camera_label.setMinimumHeight(430)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            "background:#000000; border:1px solid #2f2f2f; color:#9aa0a6; border-radius:8px;"
        )
        top_layout.addWidget(self.camera_label)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Monitoring")
        self.stop_btn = QPushButton("Stop Monitoring")
        self.refresh_snapshots_btn = QPushButton("Refresh Snapshots")
        self.stop_btn.setEnabled(False)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.refresh_snapshots_btn)
        controls.addStretch()
        top_layout.addLayout(controls)

        noise_row = QHBoxLayout()
        self.noise_label = QLabel("Noise level: 0%")
        self.noise_bar = QProgressBar()
        self.noise_bar.setRange(0, 100)
        self.noise_bar.setValue(0)
        self.noise_bar.setFormat("%p%")

        self.threshold_label = QLabel(f"Threshold: {self.noise_threshold}%")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(self.noise_threshold)

        noise_row.addWidget(self.noise_label, 1)
        noise_row.addWidget(self.noise_bar, 2)
        noise_row.addWidget(self.threshold_label, 1)
        noise_row.addWidget(self.threshold_slider, 2)
        top_layout.addLayout(noise_row)

        main.addWidget(top_card)

        lists_card = QFrame()
        lists_card.setObjectName("Card")
        lists_layout = QVBoxLayout(lists_card)
        lists_layout.setContentsMargins(12, 12, 12, 12)
        lists_layout.setSpacing(10)

        lists = QHBoxLayout()
        lists.setSpacing(8)

        people_col = QVBoxLayout()
        people_col.addWidget(QLabel("Detected Person List (Live)"))
        self.people_list = QListWidget()
        people_col.addWidget(self.people_list)

        alerts_col = QVBoxLayout()
        alerts_col.addWidget(QLabel("Misbehavior Alerts"))
        self.alerts_list = QListWidget()
        alerts_col.addWidget(self.alerts_list)

        shots_col = QVBoxLayout()
        shots_col.addWidget(QLabel("All Snapshots (Date, Time)"))
        self.snapshots_list = QListWidget()
        shots_col.addWidget(self.snapshots_list)

        lists.addLayout(people_col)
        lists.addLayout(alerts_col)
        lists.addLayout(shots_col)
        lists_layout.addLayout(lists)
        main.addWidget(lists_card)

        self.status_label = QLabel("Status: Idle")
        main.addWidget(self.status_label)

        self.setLayout(main)

        self.setStyleSheet(
            """
            QWidget {
                background:#121212;
                color:#ffffff;
                font-size:14px;
            }
            QFrame#Card {
                background:#1a1a1a;
                border:1px solid #2a2a2a;
                border-radius:10px;
            }
            QLabel {
                background: transparent;
                color: #ffffff;
            }
            QPushButton {
                background:#1f1f1f;
                border:1px solid #333333;
                padding:8px 12px;
                border-radius:6px;
                color:#ffffff;
            }
            QPushButton:hover { background:#2f2f2f; }
            QPushButton:disabled {
                background:#171717;
                color:#8f8f8f;
            }
            QListWidget, QProgressBar, QSlider {
                background:#161616;
                border:1px solid #2f2f2f;
                border-radius:6px;
                color:#ffffff;
            }
            QProgressBar::chunk {
                background:#2f9bff;
                border-radius:4px;
            }
            """
        )

        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.refresh_snapshots_btn.clicked.connect(self.refresh_snapshot_list)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.snapshots_list.itemDoubleClicked.connect(self.open_snapshot)

    def on_threshold_change(self, value):
        self.noise_threshold = value
        self.threshold_label.setText(f"Threshold: {value}%")

    def start_monitoring(self):
        if self.worker is not None:
            self.status_label.setText("Status: Monitoring is already running")
            return

        try:
            self.cam_manager.start_all()
        except Exception as exc:
            self.status_label.setText(f"Status: {exc}")
            return

        if self.cam_manager.workers:
            self.worker = self.cam_manager.workers[0]
            self.worker.frame_ready.connect(self.update_frame)
            self.status_label.setText("Status: Monitoring started")
        else:
            self.status_label.setText("Status: No camera configured in settings.json")
            return

        self._start_audio_capture()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_monitoring(self):
        self.cam_manager.stop_all()
        self.worker = None

        if self.audio_source:
            self.audio_source.stop()
            self.audio_source = None
            self.audio_device = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Monitoring stopped")

    def _start_audio_capture(self):
        if not (QAudioFormat and QAudioSource and QMediaDevices):
            self.status_label.setText("Status: Audio API unavailable")
            return

        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.status_label.setText("Status: No microphone found")
            return

        fmt = device.preferredFormat()
        if fmt.sampleRate() <= 0:
            fmt.setSampleRate(16000)
        if fmt.channelCount() <= 0:
            fmt.setChannelCount(1)
        if fmt.sampleFormat() not in (QAudioFormat.Int16, QAudioFormat.Float):
            fmt.setSampleFormat(QAudioFormat.Int16)

        self.audio_source = QAudioSource(device, fmt, self)
        self.audio_device = self.audio_source.start()

        if self.audio_device:
            self.audio_device.readyRead.connect(self._process_audio_data)
            self.status_label.setText("Status: Monitoring started (camera + microphone)")
        else:
            self.status_label.setText("Status: Microphone stream failed to start")

    def _process_audio_data(self):
        if not self.audio_device:
            return

        raw = self.audio_device.readAll()
        data = bytes(raw)
        if not data:
            return

        sample_format = self.audio_source.format().sampleFormat()
        if sample_format == QAudioFormat.Float:
            samples = np.frombuffer(data, dtype=np.float32)
            if samples.size == 0:
                return
            rms = float(np.sqrt(np.mean(np.square(samples))))
            level = int(min(100, max(0.0, rms * 180)))
        else:
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size == 0:
                return
            rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            level = int(min(100, max(0.0, (rms / 32768.0) * 220)))

        if samples.size == 0:
            return

        self.noise_level = level
        self.noise_label.setText(f"Noise level: {level}%")
        self.noise_bar.setValue(level)

        if level >= self.noise_threshold:
            self.auto_snapshot("noise")

    def update_frame(self, image, name, emotion, fps, count, popup_msg):
        self.current_frame = image
        self.camera_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        if name != "Unknown" and emotion in self.alert_emotions:
            now_ts = datetime.now().timestamp()
            prev_ts = self.last_alert_by_person.get(name, 0.0)
            if (now_ts - prev_ts) < self.alert_cooldown_sec:
                return
            self.last_alert_by_person[name] = now_ts

            stamp = datetime.now().strftime("%H:%M:%S")
            self.alerts_list.insertItem(0, f"[{stamp}] {name} -> {emotion}")

            while self.alerts_list.count() > 80:
                self.alerts_list.takeItem(self.alerts_list.count() - 1)

            self.auto_snapshot("misbehavior")

    def refresh_people_list(self):
        people = [row["name"] for row in get_all_locations() if row.get("live")]
        people = sorted(set(people))

        self.people_list.clear()
        if not people:
            self.people_list.addItem("No live person detected")
            return

        for person in people:
            self.people_list.addItem(person)

    def auto_snapshot(self, reason):
        if self.current_frame is None:
            return

        now_ts = datetime.now().timestamp()
        if (now_ts - self.last_snapshot_time) < self.snapshot_cooldown_sec:
            return

        folder = str(NOISE_ALERTS_DIR)
        os.makedirs(folder, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{reason}_{stamp}.png")

        if isinstance(self.current_frame, QImage):
            self.current_frame.save(path)
            self.last_snapshot_time = now_ts
            self.status_label.setText(f"Status: Snapshot saved ({reason})")
            self.refresh_snapshot_list()

    def refresh_snapshot_list(self):
        folder = str(NOISE_ALERTS_DIR)
        os.makedirs(folder, exist_ok=True)

        files = []
        for file_name in os.listdir(folder):
            if not file_name.lower().endswith(".png"):
                continue

            full = os.path.join(folder, file_name)
            base = os.path.splitext(file_name)[0]

            reason = "unknown"
            raw_stamp = ""
            if "_" in base:
                reason, raw_stamp = base.split("_", 1)

            date_txt = "----/--/--"
            time_txt = "--:--:--"
            try:
                dt = datetime.strptime(raw_stamp, "%Y%m%d_%H%M%S")
                date_txt = dt.strftime("%d-%m-%Y")
                time_txt = dt.strftime("%H:%M:%S")
            except Exception:
                pass

            files.append((full, date_txt, time_txt, reason))

        files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

        self.snapshots_list.clear()
        if not files:
            self.snapshots_list.addItem("No snapshots saved")
            return

        for full, date_txt, time_txt, reason in files:
            item_text = f"{date_txt}, {time_txt} | {reason}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, full)
            self.snapshots_list.addItem(item)

    def open_snapshot(self, item):
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        try:
            os.startfile(path)
        except Exception:
            self.status_label.setText("Status: Could not open snapshot file")

    def closeEvent(self, event):
        self.stop_monitoring()
        super().closeEvent(event)

