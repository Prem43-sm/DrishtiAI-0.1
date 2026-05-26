from pathlib import Path
from threading import Lock

import cv2
import face_recognition
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QTableWidget, QTableWidgetItem, QComboBox
)
from PySide6.QtCore import QThread, QTimer, Signal
from core.project_paths import ATTENDANCE_DIR, KNOWN_FACES_DIR, ensure_runtime_layout
from features.student_records_db import list_student_records
from features.tracking.live_tracker import get_all_locations, get_camera_presence, get_location, update_location
from gui.camera_backend import open_camera_capture, scan_camera_ids
from gui.face_memory import FaceMemory
from gui.settings_manager import SettingsManager


RECOGNITION_LOCK = Lock()


class TrackingCameraWorker(QThread):
    detected = Signal(str, str, str)
    status = Signal(str)

    def __init__(self, camera_id, camera_name, class_lookup, process_every=6):
        super().__init__()
        self.camera_id = int(camera_id)
        self.camera_name = str(camera_name)
        self.class_lookup = dict(class_lookup)
        self.process_every = max(1, int(process_every))
        self.running = True
        self.memory = FaceMemory.get_instance()

    def run(self):
        cap = None
        try:
            cap, backend_name, _ = open_camera_capture(self.camera_id)
            if cap is None:
                self.status.emit(f"{self.camera_name}: open failed")
                return

            self.status.emit(f"{self.camera_name}: tracking via {backend_name}")
            frame_count = 0

            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.msleep(40)
                    continue

                frame_count += 1
                if frame_count % self.process_every != 0:
                    continue

                try:
                    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    rgb = rgb.copy()
                except Exception as exc:
                    self.status.emit(f"{self.camera_name}: frame error - {exc}")
                    continue

                try:
                    with RECOGNITION_LOCK:
                        locations = face_recognition.face_locations(rgb, model="hog")
                        encodings = face_recognition.face_encodings(rgb, locations)
                        matches = [self.memory.match_face(encoding) for encoding in encodings]
                except Exception as exc:
                    self.status.emit(f"{self.camera_name}: recognition error - {exc}")
                    continue

                for match in matches:
                    name = str(match.get("name", "")).strip()
                    if not name or name == "Unknown":
                        continue
                    class_name = self.class_lookup.get(name, "Unknown")
                    self.detected.emit(name, self.camera_name, class_name)
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self.running = False
        if not self.wait(2500):
            self.terminate()
            self.wait(1000)


class TrackingPage(QWidget):
    def __init__(self):
        super().__init__()

        ensure_runtime_layout()
        self.known_faces_dir = Path(KNOWN_FACES_DIR)
        self.attendance_dir = Path(ATTENDANCE_DIR)
        self.settings = SettingsManager()
        self.workers = {}
        self.camera_names = {}

        layout = QVBoxLayout()

        title = QLabel("Live Tracking")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # 🔍 SEARCH BAR
        search_layout = QHBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")

        self.class_input = QComboBox()
        self.class_input.setEditable(False)
        self._populate_class_options()

        self.search_btn = QPushButton("Track")
        self.start_btn = QPushButton("Start All Cameras")
        self.stop_btn = QPushButton("Stop Tracking")
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.stop_btn.setEnabled(False)

        search_layout.addWidget(self.name_input)
        search_layout.addWidget(self.class_input)
        search_layout.addWidget(self.search_btn)
        search_layout.addWidget(self.start_btn)
        search_layout.addWidget(self.stop_btn)
        search_layout.addWidget(self.refresh_btn)

        layout.addLayout(search_layout)

        # 🧾 RESULT LABEL
        self.result = QLabel("Status: ---")
        layout.addWidget(self.result)

        self.camera_presence = QLabel("Live Camera Presence: ---")
        self.camera_presence.setStyleSheet(
            "font-size:15px; font-weight:bold; padding:8px; background:#1f1f1f; border-radius:4px;"
        )
        layout.addWidget(self.camera_presence)

        # 📋 LIVE TABLE
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Presence", "Live Camera", "Class", "Last Seen"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.setLayout(layout)

        # ⏱ TIMER
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_live_data)
        self.timer.start(1000)

        self.search_btn.clicked.connect(self.search_student)
        self.start_btn.clicked.connect(self.start_tracking_cameras)
        self.stop_btn.clicked.connect(self.stop_tracking_cameras)
        self.refresh_btn.clicked.connect(self.refresh_tracking_cameras)

    def _discover_classes(self):
        classes = set()
        for row in list_student_records():
            class_name = str(row.get("class_name", "")).strip()
            if class_name:
                classes.add(class_name)
        for root in (self.known_faces_dir, self.attendance_dir):
            if not root.exists():
                continue
            for item in root.iterdir():
                if item.is_dir():
                    classes.add(item.name)
        return sorted(classes)

    def _camera_name_from_settings(self, camera_id):
        data = self.settings.load()
        for camera in data.get("cameras", []):
            if int(camera.get("id", -1)) == int(camera_id):
                return str(camera.get("name") or f"Camera {camera_id}")
        return f"Camera {camera_id}"

    def _configured_camera_ids(self):
        ids = []
        data = self.settings.load()
        for camera in data.get("cameras", []):
            try:
                camera_id = int(camera.get("id"))
            except (TypeError, ValueError):
                continue
            if camera_id not in ids:
                ids.append(camera_id)
        return ids

    def _tracking_camera_ids(self):
        configured = self._configured_camera_ids()
        if configured:
            return configured
        return scan_camera_ids(8)

    def _student_class_lookup(self):
        lookup = {}
        for row in list_student_records():
            if str(row.get("status", "Active")).lower() == "left":
                continue
            name = str(row.get("student_name", "")).strip()
            class_name = str(row.get("class_name", "")).strip()
            if name and class_name:
                lookup[name] = class_name

        if self.known_faces_dir.exists():
            for class_dir in self.known_faces_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name.startswith("_"):
                    continue
                for face_file in class_dir.glob("*.npy"):
                    lookup.setdefault(face_file.stem, class_dir.name)
        return lookup

    def _populate_class_options(self):
        current = self.class_input.currentText().strip() if hasattr(self, "class_input") else ""
        classes = self._discover_classes()

        self.class_input.blockSignals(True)
        self.class_input.clear()
        self.class_input.setEnabled(bool(classes))

        if classes:
            self.class_input.addItem("Select Class")
            self.class_input.addItems(classes)
            if current and current in classes:
                self.class_input.setCurrentText(current)
            else:
                self.class_input.setCurrentIndex(0)
        else:
            self.class_input.addItem("No Classes")

        self.class_input.blockSignals(False)

    def refresh_tracking_cameras(self):
        self.stop_tracking_cameras()
        connected = self._tracking_camera_ids()
        self.camera_names = {
            camera_id: self._camera_name_from_settings(camera_id)
            for camera_id in connected
        }
        if connected:
            names = ", ".join(self.camera_names.values())
            self.result.setText(f"Status: Found {len(connected)} camera(s): {names}")
        else:
            self.result.setText("Status: No camera detected")

    def start_tracking_cameras(self):
        self.stop_tracking_cameras()
        connected = self._tracking_camera_ids()
        if not connected:
            self.result.setText("Status: No camera detected")
            return
        if len(connected) > 4:
            connected = connected[:4]
            self.result.setText("Status: Using first 4 cameras for stable live recognition")

        FaceMemory.get_instance().reload()
        lookup = self._student_class_lookup()
        self.camera_names = {
            camera_id: self._camera_name_from_settings(camera_id)
            for camera_id in connected
        }

        for camera_id in connected:
            worker = TrackingCameraWorker(
                camera_id,
                self.camera_names[camera_id],
                lookup,
                process_every=6,
            )
            worker.detected.connect(self.on_person_detected)
            worker.status.connect(self.on_worker_status)
            worker.start()
            self.workers[camera_id] = worker

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.result.setText(f"Status: Tracking {len(self.workers)} camera(s)")

    def stop_tracking_cameras(self):
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_person_detected(self, name, camera, class_name):
        update_location(name, camera, class_name)

    def on_worker_status(self, message):
        if message:
            self.result.setText(f"Status: {message}")

    # ---------------- SEARCH MODE ----------------
    def search_student(self):
        name = self.name_input.text().strip()
        data = get_location(name)
        selected_class = self.class_input.currentText().strip()

        if data and data["live"] and selected_class not in ("", "Select Class", "No Classes"):
            if data["class"] != selected_class:
                data = None

        if data and data["live"]:
            self.result.setText(
                f"Status: Present | Camera: {data['camera']} | Class: {data['class']} | Last seen: {data['time']}"
            )
        elif data:
            self.result.setText(
                f"Status: Offline | Last camera: {data['camera']} | Last seen: {data['time']}"
            )
        else:
            self.result.setText("Status: Not Found")

    # ---------------- LIVE TABLE ----------------
    def load_live_data(self):
        selected_class = self.class_input.currentText().strip()
        data = get_all_locations()
        if selected_class not in ("", "Select Class", "No Classes"):
            data = [row for row in data if row["class"] == selected_class]
        data.sort(key=lambda row: (not row["live"], row["name"].lower()))
        self.table.setRowCount(len(data))
        self._update_camera_presence(selected_class)

        for row, student in enumerate(data):

            status = "Present" if student["live"] else "Offline"

            self.table.setItem(row, 0, QTableWidgetItem(student["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(status))
            self.table.setItem(row, 2, QTableWidgetItem(student["camera"]))
            self.table.setItem(row, 3, QTableWidgetItem(student["class"]))
            self.table.setItem(row, 4, QTableWidgetItem(student["time"]))

    def _update_camera_presence(self, selected_class):
        presence = get_camera_presence(selected_class)
        if not presence:
            self.camera_presence.setText("Live Camera Presence: No live student detected")
            return

        parts = []
        for camera, names in presence.items():
            visible_names = ", ".join(names[:4])
            if len(names) > 4:
                visible_names += f" +{len(names) - 4}"
            parts.append(f"{camera}: {visible_names}")
        self.camera_presence.setText("Live Camera Presence: " + " | ".join(parts))
