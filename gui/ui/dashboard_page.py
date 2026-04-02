import math
import os
from datetime import datetime
from functools import partial

import cv2
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.camera_backend import scan_camera_ids
from gui.camera_worker import CameraWorker
from gui.emotion_model_runtime import get_model_display_name
from gui.settings_manager import SettingsManager


class CameraScanWorker(QThread):
    completed = Signal(list, str)

    def __init__(self, max_scan_cameras, action, parent=None):
        super().__init__(parent)
        self.max_scan_cameras = int(max_scan_cameras)
        self.action = action

    def run(self):
        connected = scan_camera_ids(
            self.max_scan_cameras,
            should_stop=self.isInterruptionRequested,
        )
        self.completed.emit(connected, self.action)


class DashboardPage(QWidget):
    motion_analytics_toggled = Signal(bool)

    SINGLE_PROCESS_EVERY = 2
    ALL_PROCESS_EVERY = 10
    MAX_SCAN_CAMERAS = 12

    def __init__(self):
        super().__init__()

        self.settings = SettingsManager()
        self.workers = {}

        self.selected_mode = "single"  # "single" or "all"
        self.selected_camera_id = 0
        self.selected_camera_ids = []
        self.available_camera_ids = []
        self._scan_worker = None
        self._pending_scan_action = None

        self.current_frame = None
        self.latest_frames = {}
        self.camera_tiles = {}

        self._last_popup_msg = ""
        self._last_popup_ts = 0.0
        self._popup_box = None

        self._build_ui()
        self._set_default_selection()

    def _build_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Dashboard")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        self.view_stack = QStackedWidget()

        self.camera_frame = QLabel()
        self.camera_frame.setMinimumSize(320, 180)
        self.camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_frame.setAlignment(Qt.AlignCenter)
        self.camera_frame.setStyleSheet("background-color: black;")
        self.view_stack.addWidget(self.camera_frame)

        self.multi_scroll = QScrollArea()
        self.multi_scroll.setWidgetResizable(True)
        self.multi_host = QWidget()
        self.multi_grid = QGridLayout(self.multi_host)
        self.multi_grid.setContentsMargins(6, 6, 6, 6)
        self.multi_grid.setSpacing(8)
        self.multi_scroll.setWidget(self.multi_host)
        self.view_stack.addWidget(self.multi_scroll)

        main_layout.addWidget(self.view_stack, 1)

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

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.camera_option_btn = QPushButton("Camera Option")
        self.snapshot_btn = QPushButton("Snapshot")
        self.reload_model_btn = QPushButton("Reload Model")
        self.motion_analytics_checkbox = QCheckBox("Motion Analytics")
        self.motion_analytics_checkbox.setChecked(False)
        self.motion_analytics_checkbox.setStyleSheet(
            "QCheckBox::indicator { width: 14px; height: 14px; }"
            "QCheckBox::indicator:unchecked { background: #2f2f2f; border: 1px solid #606060; }"
            "QCheckBox::indicator:checked { background: #1db954; border: 1px solid #178f43; }"
        )

        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.camera_option_btn)
        btn_layout.addWidget(self.snapshot_btn)
        btn_layout.addWidget(self.reload_model_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.motion_analytics_checkbox)
        main_layout.addLayout(btn_layout)

        stats_layout = QHBoxLayout()
        self.attendance_count = QLabel("Today Attendance: 0")
        self.known_faces_count = QLabel("Known Faces: 0")
        stats_layout.addWidget(self.attendance_count)
        stats_layout.addWidget(self.known_faces_count)
        main_layout.addLayout(stats_layout)

        self.setLayout(main_layout)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.camera_option_btn.clicked.connect(self.select_camera_option)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.reload_model_btn.clicked.connect(self.reload_model)
        self.motion_analytics_checkbox.toggled.connect(self.motion_analytics_toggled.emit)

    def _request_camera_scan(self, action):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._pending_scan_action = action
            return

        self._scan_worker = CameraScanWorker(self.MAX_SCAN_CAMERAS, action, self)
        self._scan_worker.completed.connect(self._handle_camera_scan_completed)
        self._scan_worker.start()

    def _handle_camera_scan_completed(self, connected, action):
        worker = self._scan_worker
        self.available_camera_ids = connected
        self._scan_worker = None
        if worker is not None:
            worker.deleteLater()

        if connected:
            if self.selected_camera_id not in connected:
                self.selected_camera_id = connected[0]
            if self.selected_mode == "all":
                self.selected_camera_ids = connected[:]
            else:
                self.selected_camera_ids = [self.selected_camera_id]
        else:
            self.selected_camera_ids = [] if self.selected_mode == "all" else [self.selected_camera_id]

        self._update_camera_button_label()

        if action == "select":
            self._show_camera_option_dialog(connected)
        elif action == "start":
            self._start_camera_with_connected(connected)

        pending = self._pending_scan_action
        self._pending_scan_action = None
        if pending:
            self._request_camera_scan(pending)

    def _cleanup_scan_worker(self):
        worker = self._scan_worker
        self._scan_worker = None
        self._pending_scan_action = None
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait(10000)
        worker.deleteLater()

    def _settings_camera_name(self, camera_id):
        data = self.settings.load()
        for cam in data.get("cameras", []):
            if cam.get("id") == camera_id:
                return cam.get("name", f"Camera {camera_id}")
        return f"Camera {camera_id}"

    def _set_default_selection(self):
        configured_camera = int(self.settings.load().get("camera_index", 0))
        self.selected_camera_id = configured_camera
        self.selected_camera_ids = [configured_camera]
        self._update_camera_button_label()
        self._request_camera_scan("default")

    def _update_camera_button_label(self):
        if self.selected_mode == "all":
            self.camera_option_btn.setText(f"Camera: All ({len(self.selected_camera_ids)})")
        else:
            self.camera_option_btn.setText(f"Camera: {self.selected_camera_id}")

    def select_camera_option(self):
        self._request_camera_scan("select")

    def _show_camera_option_dialog(self, connected):
        if not connected:
            QMessageBox.warning(self, "No Camera", "No connected camera detected.")
            return

        items = [f"Camera {cam_id}" for cam_id in connected]
        items.append(f"All Cameras ({len(connected)})")

        current = (
            f"All Cameras ({len(connected)})"
            if self.selected_mode == "all"
            else f"Camera {self.selected_camera_id}"
        )
        current_index = items.index(current) if current in items else 0

        selected, ok = QInputDialog.getItem(
            self,
            "Select Camera",
            "Choose camera mode:",
            items,
            current_index,
            False,
        )
        if not ok:
            return

        if selected.startswith("All Cameras"):
            self.selected_mode = "all"
            self.selected_camera_ids = connected[:]
        else:
            self.selected_mode = "single"
            self.selected_camera_id = int(selected.split(" ")[1])
            self.selected_camera_ids = connected[:]

        self._update_camera_button_label()
        self._switch_view_mode()

        if self.workers:
            self.start_camera()

    def _switch_view_mode(self):
        if self.selected_mode == "all":
            self.view_stack.setCurrentWidget(self.multi_scroll)
            self._build_multi_grid(self.selected_camera_ids)
            self._render_all_frames()
        else:
            self.view_stack.setCurrentWidget(self.camera_frame)
            self._render_current_frame()

    def _build_multi_grid(self, camera_ids):
        while self.multi_grid.count():
            item = self.multi_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.camera_tiles.clear()

        if not camera_ids:
            note = QLabel("No camera connected")
            note.setAlignment(Qt.AlignCenter)
            self.multi_grid.addWidget(note, 0, 0)
            return

        count = len(camera_ids)
        if count == 1:
            cols = 1
        elif count <= 4:
            cols = 2
        else:
            cols = max(3, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / cols)

        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols

            tile = QLabel(f"Camera {camera_id}\nWaiting for frame...")
            tile.setAlignment(Qt.AlignCenter)
            tile.setMinimumSize(260, 160)
            tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            tile.setStyleSheet("background-color: black; color: #b0b0b0;")
            self.multi_grid.addWidget(tile, row, col)
            self.camera_tiles[camera_id] = tile

        for c in range(cols):
            self.multi_grid.setColumnStretch(c, 1)
        for r in range(rows):
            self.multi_grid.setRowStretch(r, 1)

    def start_camera(self):
        self.stop_camera()
        self.start_btn.setEnabled(False)
        self._request_camera_scan("start")

    def _start_camera_with_connected(self, connected):
        if not connected:
            QMessageBox.warning(self, "No Camera", "No connected camera detected.")
            self.start_btn.setEnabled(True)
            return

        config = self.settings.load()
        configured_process_every = max(
            1,
            int(config.get("process_frame", self.SINGLE_PROCESS_EVERY)),
        )

        if self.selected_mode == "all":
            target_ids = connected
            process_every = max(configured_process_every, self.ALL_PROCESS_EVERY)
            self.selected_camera_ids = connected[:]
            self._build_multi_grid(target_ids)
            self.view_stack.setCurrentWidget(self.multi_scroll)
        else:
            if self.selected_camera_id not in connected:
                self.selected_camera_id = connected[0]
            target_ids = [self.selected_camera_id]
            process_every = configured_process_every
            self.view_stack.setCurrentWidget(self.camera_frame)

        try:
            for cam_id in target_ids:
                worker = CameraWorker(
                    camera_id=cam_id,
                    camera_name=self._settings_camera_name(cam_id),
                    process_every_n_frames=process_every,
                )
                worker.frame_ready.connect(partial(self.update_ui, cam_id))
                worker.start()
                self.workers[cam_id] = worker
        except Exception as exc:
            self.stop_camera()
            QMessageBox.warning(self, "Camera Error", str(exc))
            return

        loaded_worker = next(iter(self.workers.values()), None)
        if loaded_worker is not None:
            self.model_label.setText(
                f"Model: {get_model_display_name(loaded_worker.model_path)}"
            )
        else:
            self.model_label.setText("Model: Loaded")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._update_camera_button_label()

        if os.path.exists("known_faces"):
            count = len(os.listdir("known_faces"))
            self.known_faces_count.setText(f"Known Faces: {count}")

    def reload_model(self):
        try:
            _, _, model_path = CameraWorker.reload_shared_model()
        except Exception as exc:
            QMessageBox.warning(self, "Reload Model", str(exc))
            return

        self.model_label.setText(f"Model: {get_model_display_name(model_path)}")

        if self.workers:
            self.start_camera()
        else:
            QMessageBox.information(self, "Reload Model", "Emotion model reloaded successfully.")

    def stop_camera(self):
        self._cleanup_scan_worker()
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()

        self.current_frame = None
        self.latest_frames.clear()
        self.camera_frame.clear()
        for tile in self.camera_tiles.values():
            tile.clear()

        self.name_label.setText("Name: ---")
        self.emotion_label.setText("Emotion: ---")
        self.fps_label.setText("FPS: 0")
        self.model_label.setText("Model: Not Loaded")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._close_attendance_popup()

    def update_ui(self, camera_id, image, name, emotion, fps, count, popup_msg):
        self.latest_frames[camera_id] = image
        self.current_frame = image

        if self.selected_mode == "all":
            self._render_camera_tile(camera_id)
        else:
            self._render_current_frame()

        self.name_label.setText(f"Name: {name}")
        self.emotion_label.setText(f"Emotion: {emotion}")
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.attendance_count.setText(f"Today Attendance: {count}")

        if popup_msg and self.selected_mode != "all":
            self.show_attendance_popup(popup_msg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.selected_mode == "all":
            self._render_all_frames()
        else:
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

    def _render_camera_tile(self, camera_id):
        tile = self.camera_tiles.get(camera_id)
        frame = self.latest_frames.get(camera_id)
        if not tile or frame is None:
            return
        tile.setPixmap(
            QPixmap.fromImage(frame).scaled(
                tile.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )

    def _render_all_frames(self):
        for camera_id in list(self.camera_tiles.keys()):
            self._render_camera_tile(camera_id)

    def show_attendance_popup(self, text):
        now = datetime.now().timestamp()
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

    def take_snapshot(self):
        if self.current_frame is None:
            return

        os.makedirs("snapshots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"snapshots/snapshot_{timestamp}.png"
        self.current_frame.save(file_path)

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
