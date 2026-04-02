import os

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QLineEdit,
    QLabel,
    QMessageBox,
    QHBoxLayout,
)

from gui.emotion_model_runtime import resolve_project_path
from gui.settings_manager import SettingsManager, default_settings


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()

        self.manager = SettingsManager()
        self.data = self.manager.load()

        layout = QVBoxLayout()
        title = QLabel("Settings")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        form = QFormLayout()

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)

        self.resolution = QComboBox()
        self.resolution.addItems(["640x480", "1280x720", "1920x1080"])

        self.fps = QComboBox()
        self.fps.addItems(["25", "30", "60"])
        self.fps.setToolTip("Caps the live camera loop. Available values: 25, 30, or 60 FPS.")

        self.process_frame = QSpinBox()
        self.process_frame.setRange(1, 20)
        self.process_frame.setToolTip(
            "Analyze one live frame group every N captured frames (higher = lighter CPU load)."
        )

        self.recognition_frames = QSpinBox()
        self.recognition_frames.setRange(1, 50)
        self.recognition_frames.setToolTip(
            "Number of frames used for face recognition (higher = more accurate, slower)"
        )

        self.emotion_frames = QSpinBox()
        self.emotion_frames.setRange(1, 50)
        self.emotion_frames.setToolTip(
            "Number of frames grouped for emotion voting (higher = steadier emotion, slower updates)"
        )

        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.1, 1.0)
        self.tolerance.setDecimals(2)
        self.tolerance.setSingleStep(0.05)

        self.auto_attendance = QCheckBox("Auto Attendance")

        self.model_path = QLineEdit()
        model_btn = QPushButton("Select Model")
        model_btn.clicked.connect(self.select_model)

        self.auto_model = QCheckBox("Auto Load Model")

        self.attendance_path = QLineEdit()
        attendance_btn = QPushButton("Select Folder")
        attendance_btn.clicked.connect(self.select_attendance)

        self.course = QLineEdit()

        self.snapshot_path = QLineEdit()
        snapshot_btn = QPushButton("Select Folder")
        snapshot_btn.clicked.connect(self.select_snapshot)

        self.auto_snapshot = QCheckBox("Auto Snapshot")

        self.theme = QComboBox()
        self.theme.addItems(["dark", "light"])

        form.addRow("Camera Index", self.camera_index)
        form.addRow("Resolution", self.resolution)
        form.addRow("FPS", self.fps)
        form.addRow("Process Frame", self.process_frame)
        form.addRow("Recognition Frames", self.recognition_frames)
        form.addRow("Emotion Frames", self.emotion_frames)
        form.addRow("Face Tolerance", self.tolerance)
        form.addRow(self.auto_attendance)

        form.addRow("Model Path", self.model_path)
        form.addRow(model_btn)
        form.addRow(self.auto_model)

        form.addRow("Attendance Folder", self.attendance_path)
        form.addRow(attendance_btn)
        form.addRow("Course Name", self.course)

        form.addRow("Snapshot Folder", self.snapshot_path)
        form.addRow(snapshot_btn)
        form.addRow(self.auto_snapshot)

        form.addRow("Theme", self.theme)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        reset_btn = QPushButton("Reset")
        save_btn.clicked.connect(self.save_settings)
        reset_btn.clicked.connect(self.reset_settings)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self._apply_to_form(self.data)

    def _apply_to_form(self, data):
        self.camera_index.setValue(int(data.get("camera_index", default_settings["camera_index"])))
        self.resolution.setCurrentText(data.get("resolution", default_settings["resolution"]))
        fps_value = str(data.get("fps", default_settings["fps"]))
        fps_index = self.fps.findText(fps_value)
        self.fps.setCurrentIndex(fps_index if fps_index >= 0 else self.fps.findText(str(default_settings["fps"])))
        self.process_frame.setValue(int(data.get("process_frame", default_settings["process_frame"])))
        self.recognition_frames.setValue(
            int(data.get("recognition_frames", default_settings["recognition_frames"]))
        )
        self.emotion_frames.setValue(
            int(data.get("emotion_frames", default_settings["emotion_frames"]))
        )
        self.tolerance.setValue(float(data.get("face_tolerance", default_settings["face_tolerance"])))
        self.auto_attendance.setChecked(bool(data.get("auto_attendance", default_settings["auto_attendance"])))

        self.model_path.setText(str(data.get("model_path", default_settings["model_path"])))
        self.auto_model.setChecked(bool(data.get("auto_load_model", default_settings["auto_load_model"])))

        self.attendance_path.setText(str(data.get("attendance_path", default_settings["attendance_path"])))
        self.course.setText(str(data.get("course_name", default_settings["course_name"])))

        self.snapshot_path.setText(str(data.get("snapshot_path", default_settings["snapshot_path"])))
        self.auto_snapshot.setChecked(bool(data.get("auto_snapshot", default_settings["auto_snapshot"])))

        theme = str(data.get("theme", default_settings["theme"]))
        self.theme.setCurrentText(theme if theme in ("dark", "light") else "dark")

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model File (*.h5)")
        if path:
            self.model_path.setText(path)

    def select_attendance(self):
        path = QFileDialog.getExistingDirectory(self, "Select Attendance Folder")
        if path:
            self.attendance_path.setText(path)

    def select_snapshot(self):
        path = QFileDialog.getExistingDirectory(self, "Select Snapshot Folder")
        if path:
            self.snapshot_path.setText(path)

    def _collect_form_data(self):
        data = self.manager.load()
        data["camera_index"] = self.camera_index.value()
        data["resolution"] = self.resolution.currentText()
        data["fps"] = int(self.fps.currentText())
        data["process_frame"] = self.process_frame.value()
        data["recognition_frames"] = self.recognition_frames.value()
        data["emotion_frames"] = self.emotion_frames.value()
        data["face_tolerance"] = round(self.tolerance.value(), 2)
        data["auto_attendance"] = self.auto_attendance.isChecked()
        data["model_path"] = self.model_path.text().strip()
        data["auto_load_model"] = self.auto_model.isChecked()
        data["attendance_path"] = self.attendance_path.text().strip()
        data["course_name"] = self.course.text().strip()
        data["snapshot_path"] = self.snapshot_path.text().strip()
        data["auto_snapshot"] = self.auto_snapshot.isChecked()
        data["theme"] = self.theme.currentText()
        return data

    def _validate(self, data):
        if not data["model_path"]:
            return False, "Model path cannot be empty."
        if not os.path.exists(resolve_project_path(data["model_path"])):
            return False, "Model file does not exist."
        if not data["attendance_path"]:
            return False, "Attendance folder cannot be empty."
        if not data["snapshot_path"]:
            return False, "Snapshot folder cannot be empty."
        if not data["course_name"]:
            return False, "Course name cannot be empty."
        return True, ""

    def save_settings(self):
        data = self._collect_form_data()
        ok, message = self._validate(data)
        if not ok:
            QMessageBox.warning(self, "Invalid Settings", message)
            return

        os.makedirs(data["attendance_path"], exist_ok=True)
        os.makedirs(data["snapshot_path"], exist_ok=True)

        self.manager.save(data)
        self.data = self.manager.load()
        QMessageBox.information(self, "Saved", "Settings saved successfully.")

    def reset_settings(self):
        self.manager.reset()
        self.data = self.manager.load()
        self._apply_to_form(self.data)
        QMessageBox.information(
            self,
            "Reset",
            "Settings reset to default values (camera list preserved).",
        )
