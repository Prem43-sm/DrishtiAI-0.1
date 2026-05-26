from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap, QDesktopServices, QCursor
from PySide6.QtCore import Qt, QUrl, QTime, QThread, Signal
import os
import json
import shutil
import sqlite3
import subprocess
import sys
import pandas as pd
import numpy as np
import cv2
import face_recognition
from datetime import datetime
from functools import partial
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.project_paths import ATTENDANCE_DIR
from core.project_paths import ANALYTICS_REPORTS_DIR
from core.project_paths import KNOWN_FACES_DIR
from core.project_paths import REPORTS_DIR
from core.project_paths import SETTINGS_FILE
from core.project_paths import SNAPSHOTS_DIR
from core.project_paths import TIMETABLE_DIR
from core.project_paths import TOOLS_DIR
from core.project_paths import ensure_runtime_layout
from features.analytics.ai_insight_engine import emotion_insights, focus_insights
from features.analytics.analytics_database import (
    analytics_database_path,
    fetch_reports,
    initialize_analytics_database,
)
from features.student_records_db import (
    add_face_data,
    delete_face_data,
    get_or_create_student_for_face,
    is_admin_role,
    list_face_logs,
    list_student_records,
    log_face_action,
    mark_student_left,
    save_student_record,
    update_student_record,
)
from gui.face_memory import FaceMemory


class ReportWorker(QThread):
    finished = Signal(bool, str)

    def run(self):
        try:
            script_path = TOOLS_DIR / "monthly_report.py"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.finished.emit(True, "Monthly report generated successfully.")
            else:
                self.finished.emit(False, result.stderr)
        except Exception as e:
            self.finished.emit(False, str(e))


class DatabasePage(QWidget):

    def __init__(self):
        super().__init__()
        self.report_worker = None
        self.current_user_role = os.environ.get("DRISHTIAI_USER_ROLE", "user")
        self.current_admin_id = os.environ.get("DRISHTIAI_ADMIN_ID", "local-admin")
        self.selected_student_record = None
        self.face_admin_buttons = []
        self.student_record_admin_buttons = []

        ensure_runtime_layout()
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        os.makedirs(TIMETABLE_DIR, exist_ok=True)

        layout = QVBoxLayout(self)

        title = QLabel("Database")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        self.tabs.addTab(self.attendance_tab(), "Attendance")
        self.tabs.addTab(self.face_db_tab(), "Student Records Center")
        self.tabs.addTab(self.reports_tab(), "Reports")
        self.tabs.addTab(self.analytics_toolbox_tab(), "AI Analytics Toolbox")
        self.tabs.addTab(self.snapshots_tab(), "Snapshots")
        self.tabs.addTab(self.system_files_tab(), "System Files")

        layout.addWidget(self.tabs)

    # ================= ATTENDANCE =================
    def attendance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        filter_layout = QHBoxLayout()

        self.class_filter = QComboBox()
        self.class_filter.setMinimumWidth(220)

        self.start_time = QTimeEdit()
        self.start_time.setTime(QTime(0, 0, 0))

        self.end_time = QTimeEdit()
        self.end_time.setTime(QTime(23, 59, 59))

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_attendance)
        self.generate_report_btn = QPushButton("Generate Monthly Report")
        self.generate_report_btn.clicked.connect(self.generate_monthly_report)

        filter_layout.addWidget(QLabel("Class"))
        filter_layout.addWidget(self.class_filter)
        filter_layout.addWidget(self.start_time)
        filter_layout.addWidget(self.end_time)
        filter_layout.addWidget(load_btn)
        filter_layout.addWidget(self.generate_report_btn)

        layout.addLayout(filter_layout)

        self.attendance_table = QTableWidget()
        layout.addWidget(self.attendance_table)

        return tab

    def generate_monthly_report(self):
        self.generate_report_btn.setEnabled(False)
        self.generate_report_btn.setText("Generating...")

        self.report_worker = ReportWorker()
        self.report_worker.finished.connect(self.on_monthly_report_done)
        self.report_worker.start()

    def on_monthly_report_done(self, success, message):
        self.generate_report_btn.setEnabled(True)
        self.generate_report_btn.setText("Generate Monthly Report")

        if success:
            QMessageBox.information(self, "Success", message)
            self.load_reports()
        else:
            QMessageBox.warning(self, "Error", message)

    def load_attendance(self):
        cls = self.class_filter.currentText().strip()
        if not cls:
            return

        folder = os.path.join(str(ATTENDANCE_DIR), cls)
        if not os.path.exists(folder):
            self.attendance_table.setRowCount(0)
            self.attendance_table.setColumnCount(0)
            return

        files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)
        if not files:
            self.attendance_table.setRowCount(0)
            self.attendance_table.setColumnCount(0)
            return

        chunks = []
        for f in files:
            try:
                chunks.append(pd.read_csv(os.path.join(folder, f)))
            except Exception:
                continue

        if not chunks:
            self.attendance_table.setRowCount(0)
            self.attendance_table.setColumnCount(0)
            return

        df = pd.concat(chunks, ignore_index=True)

        if "Time" in df.columns and not df.empty:
            start = self.start_time.time().toString("HH:mm:ss")
            end = self.end_time.time().toString("HH:mm:ss")
            df = df[(df["Time"] >= start) & (df["Time"] <= end)]

        self.attendance_table.setRowCount(len(df))
        self.attendance_table.setColumnCount(len(df.columns))
        self.attendance_table.setHorizontalHeaderLabels(df.columns)
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.attendance_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))

    # ================= FACE DATABASE =================
    def face_db_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.student_center_tabs = QTabWidget()
        self.student_center_tabs.addTab(self.student_records_tab(), "Student Records")
        self.student_center_tabs.addTab(self.face_management_tab(), "Face Management")
        if self._can_manage_faces():
            self.student_center_tabs.addTab(self.audit_logs_tab(), "Audit Logs")
        layout.addWidget(self.student_center_tabs)
        self.load_classes()
        self.load_student_records()
        return tab

    def student_records_tab(self):
        tab = QWidget()
        main = QHBoxLayout(tab)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 6, 0)
        self.class_list = QListWidget()
        self.class_list.setMaximumWidth(190)
        self.class_list.setMinimumWidth(150)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.new_class = QLineEdit()
        self.new_class.setPlaceholderText("New Class")
        self.new_class.setMaximumWidth(190)

        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self.add_class)
        delete_class_btn = QPushButton("Delete Class")
        delete_class_btn.clicked.connect(self.delete_class)
        for button in (add_class_btn, delete_class_btn):
            button.setMaximumWidth(190)

        left.addWidget(QLabel("CLASSES"))
        left.addWidget(self.class_list)
        left.addWidget(self.new_class)
        left.addWidget(add_class_btn)
        left.addWidget(delete_class_btn)

        right = QVBoxLayout()
        form = QHBoxLayout()
        self.student_year = QLineEdit(str(datetime.now().year))
        self.student_year.setPlaceholderText("Academic Year")
        self.student_semester = QLineEdit()
        self.student_semester.setPlaceholderText("Semester")
        self.student_roll = QLineEdit()
        self.student_roll.setPlaceholderText("Roll Number")
        self.student_name = QLineEdit()
        self.student_name.setPlaceholderText("Student Name")
        self.student_contact = QLineEdit()
        self.student_contact.setPlaceholderText("Contact Number")
        save_record_btn = QPushButton("Save Student Record")
        save_record_btn.clicked.connect(self.save_student_record_ui)
        edit_record_btn = QPushButton("Edit Selected")
        edit_record_btn.clicked.connect(self.edit_selected_student_record)
        edit_record_btn.setToolTip("Select a row, change the fields, then click Edit Selected.")
        left_record_btn = QPushButton("Mark Left")
        left_record_btn.clicked.connect(self.mark_selected_student_left)
        left_record_btn.setToolTip("Mark the selected student as left without deleting the record.")
        self.student_record_admin_buttons = [save_record_btn, edit_record_btn, left_record_btn]
        for button in self.student_record_admin_buttons:
            button.setEnabled(self._can_manage_faces())

        for widget in (
            self.student_year,
            self.student_semester,
            self.student_roll,
            self.student_name,
            self.student_contact,
            save_record_btn,
            edit_record_btn,
            left_record_btn,
        ):
            form.addWidget(widget)

        self.student_records_table = QTableWidget()
        self.student_records_table.setColumnCount(10)
        self.student_records_table.setHorizontalHeaderLabels(
            [
                "Academic Year",
                "Class Name",
                "Semester",
                "Roll Number",
                "Student Name",
                "Contact Number",
                "Face Status",
                "Face Preview",
                "Created Date",
                "Left",
            ]
        )
        self.student_records_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.student_records_table.itemSelectionChanged.connect(self.sync_selected_student_from_table)

        right.addLayout(form)
        right.addWidget(self.student_records_table)
        main.setStretch(0, 1)
        main.setStretch(1, 8)
        main.addLayout(left, 1)
        main.addLayout(right, 8)
        return tab

    def face_management_tab(self):
        tab = QWidget()
        main = QHBoxLayout(tab)

        self.student_list = QListWidget()
        self.student_list.itemClicked.connect(self.on_face_student_selected)
        self.student_count = QLabel("Total Students: 0")

        left = QVBoxLayout()
        left.addWidget(QLabel("STUDENTS"))
        left.addWidget(self.student_list)
        left.addWidget(self.student_count)

        right = QVBoxLayout()
        self.face_profile_label = QLabel("Select a student record")
        self.face_status_label = QLabel("Face Status: Not Registered")
        self.face_preview = QLabel("No Preview")
        self.face_preview.setFixedSize(160, 160)
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setStyleSheet("border:1px solid #333; background:#111;")

        buttons = QHBoxLayout()
        register_btn = QPushButton("Register Face")
        register_btn.clicked.connect(self.add_student)
        capture_btn = QPushButton("Capture Webcam")
        capture_btn.clicked.connect(self.capture_student_face)
        retrain_btn = QPushButton("Retrain")
        retrain_btn.clicked.connect(self.retrain_selected_face)
        delete_btn = QPushButton("Delete Face Data")
        delete_btn.clicked.connect(self.delete_student)

        self.face_admin_buttons = [register_btn, capture_btn, retrain_btn, delete_btn]
        for button in self.face_admin_buttons:
            button.setEnabled(self._can_manage_faces())
            buttons.addWidget(button)

        self.face_gallery = QListWidget()
        self.face_gallery.setToolTip("Face sample gallery. Use Register Face to upload images.")

        right.addWidget(self.face_profile_label)
        right.addWidget(self.face_status_label)
        right.addWidget(self.face_preview)
        right.addLayout(buttons)
        right.addWidget(QLabel("Face Preview Gallery"))
        right.addWidget(self.face_gallery)

        main.addLayout(left, 1)
        main.addLayout(right, 3)
        return tab

    def audit_logs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        reload_btn = QPushButton("Reload Logs")
        reload_btn.clicked.connect(self.load_face_logs)
        self.face_logs_table = QTableWidget()
        self.face_logs_table.setColumnCount(9)
        self.face_logs_table.setHorizontalHeaderLabels(
            ["Action", "Admin", "Student ID", "Roll", "Student", "Class", "Semester", "Timestamp", "Device"]
        )
        self.face_logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(reload_btn)
        layout.addWidget(self.face_logs_table)
        return tab

    def _can_manage_faces(self):
        return is_admin_role(self.current_user_role)

    def set_current_user(self, user_id, role):
        self.current_admin_id = str(user_id or "local-admin").strip() or "local-admin"
        self.current_user_role = str(role or "user").strip().lower()
        can_manage = self._can_manage_faces()
        for button in getattr(self, "face_admin_buttons", []):
            button.setEnabled(can_manage)
        for button in getattr(self, "student_record_admin_buttons", []):
            button.setEnabled(can_manage)
        if hasattr(self, "student_center_tabs"):
            has_audit_tab = any(
                self.student_center_tabs.tabText(index) == "Audit Logs"
                for index in range(self.student_center_tabs.count())
            )
            if can_manage and not has_audit_tab:
                self.student_center_tabs.addTab(self.audit_logs_tab(), "Audit Logs")
            elif not can_manage and has_audit_tab:
                for index in range(self.student_center_tabs.count()):
                    if self.student_center_tabs.tabText(index) == "Audit Logs":
                        self.student_center_tabs.removeTab(index)
                        break
        self.load_student_records()

    def _require_face_admin(self):
        if self._can_manage_faces():
            return True
        QMessageBox.warning(self, "Permission Denied", "Only HOD/Admin can manage face data.")
        return False

    def _require_student_admin(self):
        if self._can_manage_faces():
            return True
        QMessageBox.warning(self, "Permission Denied", "Only HOD/Admin can modify student records.")
        return False

    def load_classes(self):
        current_combo = self.class_filter.currentText() if hasattr(self, "class_filter") else ""
        current_list = self.class_list.currentItem().text() if self.class_list.currentItem() else ""

        classes = []
        self.class_list.clear()
        if hasattr(self, "class_filter"):
            self.class_filter.clear()

        base = str(KNOWN_FACES_DIR)
        if not os.path.exists(base):
            return

        for item in sorted(os.listdir(base)):
            full = os.path.join(base, item)
            if os.path.isdir(full) and not item.startswith("_"):
                classes.append(item)
                self.class_list.addItem(item)
                if hasattr(self, "class_filter"):
                    self.class_filter.addItem(item)
                tt_path = os.path.join(str(TIMETABLE_DIR), f"{item}.json")
                if not os.path.exists(tt_path):
                    with open(tt_path, "w", encoding="utf-8") as f:
                        json.dump({"days": {}}, f, indent=4)

        # restore selections
        if current_combo and current_combo in classes and hasattr(self, "class_filter"):
            self.class_filter.setCurrentText(current_combo)
        if current_list and current_list in classes:
            matches = self.class_list.findItems(current_list, Qt.MatchExactly)
            if matches:
                self.class_list.setCurrentItem(matches[0])

    def add_class(self):
        name = self.new_class.text().strip()
        if not name:
            return

        os.makedirs(os.path.join(str(KNOWN_FACES_DIR), name), exist_ok=True)
        os.makedirs(os.path.join(str(ATTENDANCE_DIR), name), exist_ok=True)
        tt_path = os.path.join(str(TIMETABLE_DIR), f"{name}.json")
        if not os.path.exists(tt_path):
            with open(tt_path, "w", encoding="utf-8") as f:
                json.dump({"days": {}}, f, indent=4)
        self.new_class.clear()
        self.load_classes()

    def delete_class(self):
        item = self.class_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Select a class first")
            return

        cls = item.text()
        reply = QMessageBox.question(
            self,
            "Delete Class",
            f"Delete class '{cls}' from Face, Attendance and Timetable?",
        )
        if reply != QMessageBox.Yes:
            return

        face_path = os.path.join(str(KNOWN_FACES_DIR), cls)
        attendance_path = os.path.join(str(ATTENDANCE_DIR), cls)
        tt_path = os.path.join(str(TIMETABLE_DIR), f"{cls}.json")

        if os.path.exists(face_path):
            shutil.rmtree(face_path, ignore_errors=True)
        if os.path.exists(attendance_path):
            shutil.rmtree(attendance_path, ignore_errors=True)
        if os.path.exists(tt_path):
            os.remove(tt_path)

        if hasattr(self, "current_class") and self.current_class == cls:
            delattr(self, "current_class")
            if hasattr(self, "student_list"):
                self.student_list.clear()
            if hasattr(self, "student_count"):
                self.student_count.setText("Total Students: 0")

        self.load_classes()
        self.load_student_records()

    def load_students(self, item):
        self.current_class = item.text()
        self.load_student_records()

    def on_class_selected(self, item):
        self.current_class = item.text()
        class_name, semester = self._split_class_semester(self.current_class)
        self.student_semester.setText(semester)
        self.load_student_records()

    def _split_class_semester(self, class_name):
        text = str(class_name or "").strip()
        lower = text.lower()
        if "sem" not in lower:
            return text, ""
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            return text, ""
        number = "".join(ch for ch in parts[1] if ch.isdigit())
        if not number:
            return parts[0], parts[1]
        number_value = int(number)
        suffix = "th"
        if number_value % 100 not in (11, 12, 13):
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(number_value % 10, "th")
        return parts[0], f"{number_value}{suffix}"

    def save_student_record_ui(self):
        if not self._require_student_admin():
            return
        cls = self.class_list.currentItem().text() if self.class_list.currentItem() else getattr(self, "current_class", "")
        if not cls:
            QMessageBox.warning(self, "Error", "Select a class first")
            return
        if not self.student_roll.text().strip() or not self.student_name.text().strip():
            QMessageBox.warning(self, "Error", "Roll number and student name are required")
            return

        _, fallback_semester = self._split_class_semester(cls)
        student_id = save_student_record(
            {
                "academic_year": self.student_year.text().strip(),
                "class_name": cls,
                "semester": self.student_semester.text().strip() or fallback_semester,
                "roll_number": self.student_roll.text().strip(),
                "student_name": self.student_name.text().strip(),
                "contact_number": self.student_contact.text().strip(),
            }
        )
        log_face_action("student_create", int(student_id), self.current_admin_id, "student-records")
        self.load_student_records()
        QMessageBox.information(self, "Saved", "Student record saved.")

    def edit_selected_student_record(self):
        if not self._require_student_admin():
            return
        if not self.selected_student_record:
            QMessageBox.warning(self, "Error", "Select a student row first")
            return
        cls = self.class_list.currentItem().text() if self.class_list.currentItem() else getattr(self, "current_class", "")
        if not cls:
            QMessageBox.warning(self, "Error", "Select a class first")
            return
        if not self.student_roll.text().strip() or not self.student_name.text().strip():
            QMessageBox.warning(self, "Error", "Roll number and student name are required")
            return

        _, fallback_semester = self._split_class_semester(cls)
        old_row = dict(self.selected_student_record)
        new_student_name = self.student_name.text().strip()
        update_student_record(
            int(old_row["id"]),
            {
                "academic_year": self.student_year.text().strip(),
                "class_name": cls,
                "semester": self.student_semester.text().strip() or fallback_semester,
                "roll_number": self.student_roll.text().strip(),
                "student_name": new_student_name,
                "contact_number": self.student_contact.text().strip(),
            },
        )
        self._move_legacy_face_file(old_row, cls, new_student_name)
        self._sync_attendance_student_identity(old_row, cls, new_student_name)
        log_face_action("student_update", int(old_row["id"]), self.current_admin_id, "student-records")
        self.load_student_records()
        QMessageBox.information(self, "Updated", "Selected student record updated.")

    def mark_selected_student_left(self):
        if not self._require_student_admin():
            return
        if not self.selected_student_record:
            QMessageBox.warning(self, "Error", "Select a student row first")
            return
        reply = QMessageBox.question(
            self,
            "Mark Left",
            "Mark selected student as left? The record and face data will stay saved.",
        )
        if reply != QMessageBox.Yes:
            return
        mark_student_left(int(self.selected_student_record["id"]))
        log_face_action("student_left", int(self.selected_student_record["id"]), self.current_admin_id, "student-records")
        self.load_student_records()
        QMessageBox.information(self, "Updated", "Student marked as left.")

    def _move_legacy_face_file(self, old_row, new_class_name, new_student_name):
        old_path = os.path.join(
            str(KNOWN_FACES_DIR),
            str(old_row.get("class_name", "")),
            f"{old_row.get('student_name', '')}.npy",
        )
        new_dir = os.path.join(str(KNOWN_FACES_DIR), str(new_class_name))
        new_path = os.path.join(new_dir, f"{new_student_name}.npy")
        if old_path == new_path or not os.path.exists(old_path):
            return
        os.makedirs(new_dir, exist_ok=True)
        shutil.move(old_path, new_path)
        FaceMemory.get_instance().reload()

    def _sync_attendance_student_identity(self, old_row, new_class_name, new_student_name):
        old_class = str(old_row.get("class_name", "")).strip()
        old_name = str(old_row.get("student_name", "")).strip()
        if not old_class or not old_name:
            return
        source_dir = os.path.join(str(ATTENDANCE_DIR), old_class)
        if not os.path.exists(source_dir):
            return

        target_dir = os.path.join(str(ATTENDANCE_DIR), str(new_class_name))
        os.makedirs(target_dir, exist_ok=True)
        class_changed = old_class != str(new_class_name)

        for file_name in sorted(os.listdir(source_dir)):
            if not file_name.endswith(".csv"):
                continue
            source_path = os.path.join(source_dir, file_name)
            try:
                data = pd.read_csv(source_path)
            except Exception:
                continue
            if "Name" not in data.columns:
                continue
            mask = data["Name"].astype(str).str.strip() == old_name
            if not mask.any():
                continue

            matched = data.loc[mask].copy()
            data.loc[mask, "Name"] = new_student_name
            if "Class" in data.columns:
                data.loc[mask, "Class"] = new_class_name
                matched["Class"] = new_class_name
            matched["Name"] = new_student_name

            if class_changed:
                target_path = os.path.join(target_dir, file_name)
                remaining = data.loc[~mask].copy()
                remaining.to_csv(source_path, index=False)
                if os.path.exists(target_path):
                    try:
                        target_data = pd.read_csv(target_path)
                    except Exception:
                        target_data = pd.DataFrame(columns=data.columns)
                    target_data = pd.concat([target_data, matched], ignore_index=True)
                else:
                    target_data = matched
                target_data.to_csv(target_path, index=False)
            else:
                data.to_csv(source_path, index=False)

    def _sync_legacy_faces_to_records(self, class_name=None):
        base = str(KNOWN_FACES_DIR)
        if not os.path.exists(base):
            return
        class_names = [class_name] if class_name else [
            item
            for item in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, item)) and not item.startswith("_")
        ]
        for cls in class_names:
            if not cls:
                continue
            _, semester = self._split_class_semester(cls)
            class_path = os.path.join(base, cls)
            if not os.path.exists(class_path):
                continue
            for file_name in os.listdir(class_path):
                if file_name.endswith(".npy"):
                    student_name = file_name.replace(".npy", "")
                    student_id = get_or_create_student_for_face(cls, semester, student_name)
                    with sqlite3.connect(str(analytics_database_path())) as conn:
                        count = conn.execute(
                            "SELECT COUNT(*) FROM face_data WHERE student_id = ?",
                            (int(student_id),),
                        ).fetchone()[0]
                    if count == 0:
                        try:
                            embedding = np.load(os.path.join(class_path, file_name))
                            add_face_data(int(student_id), "", embedding, "legacy-migration")
                        except Exception:
                            continue

    def load_student_records(self):
        if not hasattr(self, "student_records_table"):
            return
        cls = self.class_list.currentItem().text() if self.class_list.currentItem() else getattr(self, "current_class", "")
        self._sync_legacy_faces_to_records(cls or None)
        records = list_student_records(cls or None)
        self.student_records_table.setRowCount(len(records))

        if hasattr(self, "student_list"):
            self.student_list.clear()
        for row_index, row in enumerate(records):
            face_count = int(row.get("face_count") or 0)
            status = "Registered" if self._has_legacy_or_db_face(row) else "Not Registered"
            preview = "Available" if row.get("preview_path") else ""
            values = [
                row.get("academic_year", ""),
                row.get("class_name", ""),
                row.get("semester", ""),
                row.get("roll_number", ""),
                row.get("student_name", ""),
                row.get("contact_number", ""),
                status,
                preview,
                self._date_display(row.get("created_at", "")),
                self._left_display(row),
            ]
            for column, value in enumerate(values):
                if column == 7:
                    preview_btn = QPushButton("View" if preview else "No Image")
                    preview_btn.setEnabled(bool(preview))
                    preview_btn.clicked.connect(
                        lambda checked=False, student_id=int(row["id"]): self.show_face_preview_dialog(student_id)
                    )
                    self.student_records_table.setCellWidget(row_index, column, preview_btn)
                    continue
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, int(row["id"]))
                self.student_records_table.setItem(row_index, column, item)

            if hasattr(self, "student_list"):
                list_item = QListWidgetItem(f"{row.get('roll_number', '')} - {row.get('student_name', '')} ({status})")
                list_item.setData(Qt.UserRole, int(row["id"]))
                list_item.setData(Qt.UserRole + 1, row)
                self.student_list.addItem(list_item)

        if hasattr(self, "student_count"):
            self.student_count.setText(f"Total Students: {len(records)}")
        if hasattr(self, "face_logs_table"):
            self.load_face_logs()

    def _has_legacy_or_db_face(self, row):
        if int(row.get("face_count") or 0) > 0:
            return True
        path = os.path.join(str(KNOWN_FACES_DIR), str(row.get("class_name", "")), f"{row.get('student_name', '')}.npy")
        return os.path.exists(path)

    def _left_display(self, row):
        if str(row.get("status", "")).lower() != "left":
            return "-"
        left_at = str(row.get("left_at") or row.get("updated_at") or "").strip()
        date_part = left_at.split("T", 1)[0] if left_at else ""
        return f"Left ({date_part})" if date_part else "Left"

    def _date_display(self, value):
        text = str(value or "").strip()
        return text.split("T", 1)[0] if text else ""

    def sync_selected_student_from_table(self):
        selected = self.student_records_table.selectedItems() if hasattr(self, "student_records_table") else []
        if not selected:
            return
        student_id = int(selected[0].data(Qt.UserRole))
        for row in list_student_records():
            if int(row["id"]) == student_id:
                self._set_selected_student(row)
                return

    def on_face_student_selected(self, item):
        row = item.data(Qt.UserRole + 1)
        if row:
            self._set_selected_student(row)

    def _set_selected_student(self, row):
        self.selected_student_record = row
        self.student_year.setText(str(row.get("academic_year", "")))
        self.student_semester.setText(str(row.get("semester", "")))
        self.student_roll.setText(str(row.get("roll_number", "")))
        self.student_name.setText(str(row.get("student_name", "")))
        self.student_contact.setText(str(row.get("contact_number", "")))
        status = "Registered" if self._has_legacy_or_db_face(row) else "Not Registered"
        self.face_profile_label.setText(
            f"{row.get('student_name', '')} | Roll: {row.get('roll_number', '')} | {row.get('class_name', '')}"
        )
        self.face_status_label.setText(f"Face Status: {status}")
        self.load_face_gallery(int(row["id"]))

    def load_face_gallery(self, student_id):
        if not hasattr(self, "face_gallery"):
            return
        self.face_gallery.clear()
        preview_path = ""
        with sqlite3.connect(str(analytics_database_path())) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, image_path, created_at FROM face_data WHERE student_id = ? ORDER BY created_at DESC",
                (int(student_id),),
            ).fetchall()
        for row in rows:
            self.face_gallery.addItem(f"Sample {row['id']} | {row['created_at']}")
            if not preview_path and row["image_path"]:
                preview_path = row["image_path"]
        if preview_path and os.path.exists(preview_path):
            self.face_preview.setPixmap(QPixmap(preview_path).scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.face_preview.setPixmap(QPixmap())
            self.face_preview.setText("No Preview")

    def show_face_preview_dialog(self, student_id):
        image_path = self._face_preview_path(student_id)
        if not image_path:
            QMessageBox.information(
                self,
                "Face Preview",
                "No saved face image is available for this student. The face may only have an embedding.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Face Preview")
        layout = QVBoxLayout(dialog)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Face Preview", "Unable to open the saved face image.")
            return
        image_label.setPixmap(pixmap.scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image_label)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.resize(460, 500)
        dialog.exec()

    def _face_preview_path(self, student_id):
        with sqlite3.connect(str(analytics_database_path())) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT image_path
                FROM face_data
                WHERE student_id = ?
                  AND image_path IS NOT NULL
                  AND image_path != ''
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (int(student_id),),
            ).fetchone()
        if not row:
            return ""
        image_path = str(row["image_path"] or "")
        return image_path if image_path and os.path.exists(image_path) else ""

    def add_student(self):
        if not self._require_face_admin():
            return
        row = self.selected_student_record
        if not row:
            QMessageBox.warning(self, "Error", "Select a student record first")
            return

        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file:
            return
        self._register_face_image(row, file)

    def _register_face_image(self, row, file):
        img = cv2.imread(file)
        if img is None:
            QMessageBox.warning(self, "Error", "Unable to read image")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        loc = face_recognition.face_locations(rgb)
        if not loc:
            QMessageBox.warning(self, "Error", "No face found")
            return

        enc = face_recognition.face_encodings(rgb, loc)[0]

        class_name = str(row.get("class_name", ""))
        student_name = str(row.get("student_name", ""))
        face_dir = os.path.join(str(KNOWN_FACES_DIR), class_name)
        sample_dir = os.path.join(face_dir, "_face_images", self._safe_file_stem(student_name))
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(face_dir, exist_ok=True)
        image_path = os.path.join(sample_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(image_path, img)

        np.save(os.path.join(face_dir, f"{student_name}.npy"), enc)
        add_face_data(int(row["id"]), image_path, enc, self.current_admin_id)
        FaceMemory.get_instance().reload()
        self.load_student_records()
        self._set_selected_student(dict(row))

        QMessageBox.information(self, "Success", "Face data registered.")

    def capture_student_face(self):
        if not self._require_face_admin():
            return
        row = self.selected_student_record
        if not row:
            QMessageBox.warning(self, "Error", "Select a student record first")
            return

        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "Error", "Unable to capture from webcam")
            return

        class_name = str(row.get("class_name", ""))
        student_name = str(row.get("student_name", ""))
        sample_dir = os.path.join(str(KNOWN_FACES_DIR), class_name, "_face_images", self._safe_file_stem(student_name))
        os.makedirs(sample_dir, exist_ok=True)
        image_path = os.path.join(sample_dir, f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(image_path, frame)
        self._register_face_image(row, image_path)

    def retrain_selected_face(self):
        if not self._require_face_admin():
            return
        row = self.selected_student_record
        if not row:
            QMessageBox.warning(self, "Error", "Select a student record first")
            return
        log_face_action("retrain", int(row["id"]), self.current_admin_id)
        FaceMemory.get_instance().reload()
        self.load_face_logs()
        QMessageBox.information(self, "Done", "Face embeddings reloaded for recognition.")

    def _safe_file_stem(self, value):
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value)).strip("_") or "student"

    def delete_student(self):
        if not self._require_face_admin():
            return
        row = self.selected_student_record
        if not row:
            QMessageBox.warning(self, "Error", "Select a student record first")
            return

        path = os.path.join(str(KNOWN_FACES_DIR), str(row.get("class_name", "")), f"{row.get('student_name', '')}.npy")

        reply = QMessageBox.question(self, "Delete", "Delete only face data? Academic record will remain safe.")
        if reply == QMessageBox.Yes:
            for image_path in delete_face_data(int(row["id"]), self.current_admin_id):
                if os.path.exists(image_path):
                    os.remove(image_path)
            if os.path.exists(path):
                os.remove(path)
            FaceMemory.get_instance().reload()
            self.load_student_records()
            self._set_selected_student(dict(row))

    def load_face_logs(self):
        if not hasattr(self, "face_logs_table"):
            return
        logs = list_face_logs()
        self.face_logs_table.setRowCount(len(logs))
        for row_index, row in enumerate(logs):
            values = [
                row.get("action_type", ""),
                row.get("admin_id", ""),
                row.get("student_id", ""),
                row.get("roll_number", ""),
                row.get("student_name", ""),
                row.get("class_name", ""),
                row.get("semester", ""),
                row.get("timestamp", ""),
                row.get("device_info", ""),
            ]
            for column, value in enumerate(values):
                self.face_logs_table.setItem(row_index, column, QTableWidgetItem(str(value)))

    def showEvent(self, event):
        self.load_classes()
        self.load_student_records()
        super().showEvent(event)

    # ================= REPORTS =================
    def reports_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        btn_layout = QHBoxLayout()

        view_btn = QPushButton("👁 View")
        delete_btn = QPushButton("❌ Delete")

        view_btn.clicked.connect(self.view_report)
        delete_btn.clicked.connect(self.delete_report)

        btn_layout.addWidget(view_btn)
        btn_layout.addWidget(delete_btn)

        layout.addLayout(btn_layout)

        self.report_list = QListWidget()
        layout.addWidget(self.report_list)

        self.load_reports()
        return tab

    def load_reports(self):
        self.report_list.clear()
        folder = str(REPORTS_DIR)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                self.report_list.addItem(file)

    # ================= AI ANALYTICS TOOLBOX =================
    def analytics_toolbox_tab(self):
        initialize_analytics_database()
        tab = QWidget()
        main = QHBoxLayout(tab)

        toolbox = QVBoxLayout()
        toolbox.addWidget(QLabel("AI Analytics Toolbox"))

        buttons = [
            ("Emotion Reports", lambda: self.load_analytics_panel("emotion_reports")),
            ("Focus Reports", lambda: self.load_analytics_panel("focus_reports")),
            ("Monthly Analytics", lambda: self.load_analytics_panel("monthly_analytics")),
            ("Export Reports", self.open_analytics_reports_folder),
            ("Student Comparison", lambda: self.load_analytics_panel("student_comparison")),
            ("AI Insights Dashboard", lambda: self.load_analytics_panel("ai_insights")),
        ]
        for label, handler in buttons:
            btn = QPushButton(label)
            btn.setMinimumHeight(38)
            btn.clicked.connect(handler)
            toolbox.addWidget(btn)
        toolbox.addStretch()

        content = QVBoxLayout()
        self.analytics_title = QLabel("Select a toolbox item")
        self.analytics_title.setStyleSheet("font-size:16px; font-weight:bold;")
        content.addWidget(self.analytics_title)

        self.analytics_chart_fig = Figure(figsize=(5, 2.8), tight_layout=True)
        self.analytics_chart_fig.patch.set_facecolor("#0b0f14")
        self.analytics_chart = FigureCanvas(self.analytics_chart_fig)
        content.addWidget(self.analytics_chart)

        self.analytics_insights = QLabel("Database: " + str(analytics_database_path()))
        self.analytics_insights.setWordWrap(True)
        self.analytics_insights.setStyleSheet("background:#101419; border:1px solid #28313d; padding:8px;")
        content.addWidget(self.analytics_insights)

        self.analytics_table = QTableWidget()
        content.addWidget(self.analytics_table, 1)

        main.addLayout(toolbox, 1)
        main.addLayout(content, 4)
        self.load_analytics_panel("emotion_reports")
        return tab

    def load_analytics_panel(self, panel_name):
        if panel_name in ("emotion_reports", "monthly_analytics"):
            rows = fetch_reports("emotion_reports", limit=250)
            self.analytics_title.setText("Emotion Reports" if panel_name == "emotion_reports" else "Monthly Analytics")
            self.populate_analytics_table(rows)
            self.draw_emotion_report_chart(rows)
            self.analytics_insights.setText(self.build_emotion_insight_text(rows))
        elif panel_name in ("focus_reports", "student_comparison"):
            rows = fetch_reports("focus_reports", limit=250)
            self.analytics_title.setText("Focus Reports" if panel_name == "focus_reports" else "Student Comparison")
            self.populate_analytics_table(rows)
            self.draw_focus_report_chart(rows, comparison=(panel_name == "student_comparison"))
            self.analytics_insights.setText(self.build_focus_insight_text(rows))
        elif panel_name == "ai_insights":
            emotion_rows = fetch_reports("emotion_reports", limit=50)
            focus_rows = fetch_reports("focus_reports", limit=50)
            rows = emotion_rows + focus_rows
            self.analytics_title.setText("AI Insights Dashboard")
            self.populate_analytics_table(rows)
            self.draw_insights_chart(emotion_rows, focus_rows)
            combined = [self.build_emotion_insight_text(emotion_rows), self.build_focus_insight_text(focus_rows)]
            self.analytics_insights.setText("\n".join(text for text in combined if text.strip()))

    def populate_analytics_table(self, rows):
        if not rows:
            self.analytics_table.setRowCount(0)
            self.analytics_table.setColumnCount(0)
            return
        columns = list(rows[0].keys())
        self.analytics_table.setColumnCount(len(columns))
        self.analytics_table.setHorizontalHeaderLabels(columns)
        self.analytics_table.setRowCount(len(rows))
        self.analytics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for r, row in enumerate(rows):
            for c, column in enumerate(columns):
                self.analytics_table.setItem(r, c, QTableWidgetItem(str(row.get(column, ""))))

    def draw_emotion_report_chart(self, rows):
        self.analytics_chart_fig.clear()
        ax = self.analytics_chart_fig.add_subplot(111)
        self._style_analytics_axis(ax)
        if not rows:
            ax.text(0.5, 0.5, "No emotion reports saved", ha="center", va="center", color="#cbd5e1")
            ax.set_axis_off()
        else:
            latest = list(reversed(rows[:12]))
            labels = [str(row.get("student_id", ""))[:10] for row in latest]
            values = [float(row.get("performance_score", 0)) for row in latest]
            ax.bar(labels, values, color="#38bdf8")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Performance")
            ax.tick_params(axis="x", rotation=25)
        self.analytics_chart.draw_idle()

    def draw_focus_report_chart(self, rows, comparison=False):
        self.analytics_chart_fig.clear()
        ax = self.analytics_chart_fig.add_subplot(111)
        self._style_analytics_axis(ax)
        if not rows:
            ax.text(0.5, 0.5, "No focus reports saved", ha="center", va="center", color="#cbd5e1")
            ax.set_axis_off()
        else:
            df = pd.DataFrame(rows)
            if comparison:
                summary = df.groupby("student_id")["focus_score"].mean().sort_values(ascending=False).head(12)
                ax.bar(summary.index.astype(str), summary.values, color="#22c55e")
                ax.set_ylabel("Average Focus")
            else:
                latest = list(reversed(rows[:12]))
                ax.plot([str(r.get("student_id", ""))[:10] for r in latest], [float(r.get("focus_score", 0)) for r in latest], marker="o", color="#22c55e")
                ax.set_ylabel("Focus")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=25)
        self.analytics_chart.draw_idle()

    def draw_insights_chart(self, emotion_rows, focus_rows):
        self.analytics_chart_fig.clear()
        ax = self.analytics_chart_fig.add_subplot(111)
        self._style_analytics_axis(ax)
        emotion_avg = np.mean([float(r.get("performance_score", 0)) for r in emotion_rows]) if emotion_rows else 0
        focus_avg = np.mean([float(r.get("focus_score", 0)) for r in focus_rows]) if focus_rows else 0
        ax.bar(["Performance", "Focus"], [emotion_avg, focus_avg], color=["#38bdf8", "#22c55e"])
        ax.set_ylim(0, 100)
        self.analytics_chart.draw_idle()

    def build_emotion_insight_text(self, rows):
        if not rows:
            return "Emotion insights: No saved emotion reports yet."
        latest = rows[0]
        insights = emotion_insights(latest)
        return "Emotion insights: " + " ".join(insights)

    def build_focus_insight_text(self, rows):
        if not rows:
            return "Focus insights: No saved focus reports yet."
        latest = rows[0]
        insights = focus_insights(latest)
        return "Focus insights: " + " ".join(insights)

    def open_analytics_reports_folder(self):
        os.makedirs(str(ANALYTICS_REPORTS_DIR), exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ANALYTICS_REPORTS_DIR)))

    def _style_analytics_axis(self, ax):
        ax.set_facecolor("#0f141b")
        for spine in ax.spines.values():
            spine.set_color("#425167")
        ax.tick_params(colors="#d7dde8")
        ax.xaxis.label.set_color("#e6ecf5")
        ax.yaxis.label.set_color("#e6ecf5")

    def view_report(self):
        item = self.report_list.currentItem()
        if item:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(os.path.join(str(REPORTS_DIR), item.text()))
            )

    def delete_report(self):
        item = self.report_list.currentItem()
        if not item:
            return

        reply = QMessageBox.question(self, "Delete", "Delete selected report?")
        if reply == QMessageBox.Yes:
            target_path = os.path.join(str(REPORTS_DIR), item.text())
            if os.path.isdir(target_path):
                shutil.rmtree(target_path, ignore_errors=True)
            elif os.path.exists(target_path):
                os.remove(target_path)
            self.load_reports()

    # ================= SNAPSHOTS =================
    def snapshots_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.snap_container = QWidget()
        self.grid = QGridLayout(self.snap_container)

        scroll.setWidget(self.snap_container)
        layout.addWidget(scroll)

        self.load_snapshots()
        return tab

    def load_snapshots(self):

        folder = str(SNAPSHOTS_DIR)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not os.path.exists(folder):
            return

        row = col = 0

        for img in sorted(os.listdir(folder), reverse=True):

            path = os.path.join(folder, img)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(img)[1].lower() not in valid_exts:
                continue

            # 🕒 format date time
            try:
                dt = datetime.strptime(
                    img.replace("snapshot_", "").replace(".png", ""),
                    "%Y%m%d_%H%M%S"
                )
                time_text = dt.strftime("%d-%m-%Y  %H:%M:%S")
            except:
                time_text = img

            # 🖼 image
            thumb = QLabel()
            thumb.setFixedSize(220, 160)
            thumb.setCursor(Qt.PointingHandCursor)

            pix = QPixmap(path)
            thumb.setPixmap(
                pix.scaled(220, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # LEFT CLICK → PREVIEW
            thumb.mousePressEvent = partial(self.preview_image_event, path)

            # RIGHT CLICK → MENU
            thumb.setContextMenuPolicy(Qt.CustomContextMenu)
            thumb.customContextMenuRequested.connect(
                partial(self.snapshot_menu, path)
            )

            # 🏷 label
            lbl = QLabel(time_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:lightgray; font-size:11px;")

            # 📦 box
            box = QVBoxLayout()
            box.addWidget(thumb)
            box.addWidget(lbl)

            wrapper = QWidget()
            wrapper.setLayout(box)
            wrapper.setStyleSheet(
                "background:#1a1a1a; border-radius:10px; padding:6px;"
            )

            self.grid.addWidget(wrapper, row, col)

            col += 1
            if col == 4:
                col = 0
                row += 1

    # ================= PREVIEW =================
    def preview_image_event(self, path, event):
        if event.button() == Qt.LeftButton:
            self.preview_image(path)

    def preview_image(self, path):
        dlg = QDialog(self)
        dlg.setWindowTitle("Preview")

        lbl = QLabel()
        lbl.setPixmap(
            QPixmap(path).scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        lay = QVBoxLayout(dlg)
        lay.addWidget(lbl)

        dlg.exec()

    # ================= RIGHT CLICK MENU =================
    def snapshot_menu(self, path, pos):

        menu = QMenu()

        share = menu.addAction("📤 Share")
        delete = menu.addAction("🗑 Delete")

        action = menu.exec(QCursor.pos())

        if action == share:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

        elif action == delete:
            reply = QMessageBox.question(
                self, "Delete", "Delete this snapshot?"
            )

            if reply == QMessageBox.Yes:
                os.remove(path)
                self.load_snapshots()
    # ================= SYSTEM FILES =================
    def system_files_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(
            lambda item: QDesktopServices.openUrl(QUrl.fromLocalFile(item.text()))
        )

        layout.addWidget(self.file_list)

        file_paths = [SETTINGS_FILE]
        if TIMETABLE_DIR.exists():
            file_paths.extend(sorted(TIMETABLE_DIR.rglob("*.json")))
        if ATTENDANCE_DIR.exists():
            file_paths.extend(sorted(ATTENDANCE_DIR.rglob("*.csv")))

        for file_path in file_paths:
            self.file_list.addItem(str(file_path))

        return tab
