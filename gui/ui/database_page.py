from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap, QDesktopServices, QCursor
from PySide6.QtCore import Qt, QUrl, QTime, QThread, Signal
import os
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
import cv2
import face_recognition
from datetime import datetime
from functools import partial
from face_memory import FaceMemory


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
                self.finished.emit(True, "Monthly report generated successfully.")
            else:
                self.finished.emit(False, result.stderr)
        except Exception as e:
            self.finished.emit(False, str(e))


class DatabasePage(QWidget):

    def __init__(self):
        super().__init__()
        self.report_worker = None

        os.makedirs("known_faces", exist_ok=True)
        os.makedirs("attendance", exist_ok=True)
        os.makedirs("timetable", exist_ok=True)

        layout = QVBoxLayout(self)

        title = QLabel("Database")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        self.tabs.addTab(self.attendance_tab(), "Attendance")
        self.tabs.addTab(self.face_db_tab(), "Face Database")
        self.tabs.addTab(self.reports_tab(), "Reports")
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

        folder = os.path.join("attendance", cls)
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
        main = QHBoxLayout(tab)

        # LEFT → CLASS LIST
        left = QVBoxLayout()

        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.load_students)

        self.new_class = QLineEdit()
        self.new_class.setPlaceholderText("New Class")

        add_class_btn = QPushButton("➕ Add Class")
        add_class_btn.clicked.connect(self.add_class)
        delete_class_btn = QPushButton("Delete Class")
        delete_class_btn.clicked.connect(self.delete_class)

        left.addWidget(QLabel("CLASSES"))
        left.addWidget(self.class_list)
        left.addWidget(self.new_class)
        left.addWidget(add_class_btn)
        left.addWidget(delete_class_btn)

        # RIGHT → STUDENTS
        right = QVBoxLayout()

        self.student_name = QLineEdit()
        self.student_name.setPlaceholderText("Student Name")

        add_student_btn = QPushButton("➕ Add Student Face")
        add_student_btn.clicked.connect(self.add_student)

        self.student_list = QListWidget()

        delete_btn = QPushButton("❌ Delete Student")
        delete_btn.clicked.connect(self.delete_student)

        self.student_count = QLabel("Total Students: 0")

        right.addWidget(self.student_name)
        right.addWidget(add_student_btn)
        right.addWidget(self.student_list)
        right.addWidget(delete_btn)
        right.addWidget(self.student_count)

        main.addLayout(left, 1)
        main.addLayout(right, 2)

        self.load_classes()

        return tab

    def load_classes(self):
        current_combo = self.class_filter.currentText() if hasattr(self, "class_filter") else ""
        current_list = self.class_list.currentItem().text() if self.class_list.currentItem() else ""

        classes = []
        self.class_list.clear()
        if hasattr(self, "class_filter"):
            self.class_filter.clear()

        base = "known_faces"
        if not os.path.exists(base):
            return

        for item in sorted(os.listdir(base)):
            full = os.path.join(base, item)
            if os.path.isdir(full):
                classes.append(item)
                self.class_list.addItem(item)
                if hasattr(self, "class_filter"):
                    self.class_filter.addItem(item)
                tt_path = os.path.join("timetable", f"{item}.json")
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

        os.makedirs(f"known_faces/{name}", exist_ok=True)
        os.makedirs(f"attendance/{name}", exist_ok=True)
        tt_path = f"timetable/{name}.json"
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

        face_path = f"known_faces/{cls}"
        attendance_path = f"attendance/{cls}"
        tt_path = f"timetable/{cls}.json"

        if os.path.exists(face_path):
            shutil.rmtree(face_path, ignore_errors=True)
        if os.path.exists(attendance_path):
            shutil.rmtree(attendance_path, ignore_errors=True)
        if os.path.exists(tt_path):
            os.remove(tt_path)

        if hasattr(self, "current_class") and self.current_class == cls:
            delattr(self, "current_class")
            self.student_list.clear()
            self.student_count.setText("Total Students: 0")

        self.load_classes()

    def load_students(self, item):
        self.current_class = item.text()
        self.student_list.clear()

        path = f"known_faces/{self.current_class}"
        if not os.path.exists(path):
            return

        for file in os.listdir(path):
            if file.endswith(".npy"):
                self.student_list.addItem(file.replace(".npy", ""))

        self.student_count.setText(f"Total Students: {self.student_list.count()}")

    def add_student(self):
        if not hasattr(self, "current_class"):
            QMessageBox.warning(self, "Error", "Select a class first")
            return

        name = self.student_name.text().strip()
        if not name:
            return

        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file:
            return

        img = cv2.imread(file)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        loc = face_recognition.face_locations(rgb)
        if not loc:
            QMessageBox.warning(self, "Error", "No face found")
            return

        enc = face_recognition.face_encodings(rgb, loc)[0]

        np.save(f"known_faces/{self.current_class}/{name}.npy", enc)

        self.student_name.clear()
        self.load_students(QListWidgetItem(self.current_class))

        # 🔥 IMPORTANT FIX
        FaceMemory.get_instance().reload()

        QMessageBox.information(self, "Success", "Student Added ✅")

    def delete_student(self):
        item = self.student_list.currentItem()
        if not item:
            return

        path = f"known_faces/{self.current_class}/{item.text()}.npy"

        reply = QMessageBox.question(self, "Delete", "Delete student?")
        if reply == QMessageBox.Yes:
            os.remove(path)
            self.load_students(QListWidgetItem(self.current_class))
            FaceMemory.get_instance().reload()

    def showEvent(self, event):
        self.load_classes()
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
        folder = "reports"
        if os.path.exists(folder):
            for file in os.listdir(folder):
                self.report_list.addItem(file)

    def view_report(self):
        item = self.report_list.currentItem()
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(f"reports/{item.text()}"))

    def delete_report(self):
        item = self.report_list.currentItem()
        if not item:
            return

        reply = QMessageBox.question(self, "Delete", "Delete selected report?")
        if reply == QMessageBox.Yes:
            os.remove(f"reports/{item.text()}")
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

        folder = "snapshots"
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

        for file in os.listdir():
            if file.endswith(".json") or file.endswith(".csv"):
                self.file_list.addItem(file)

        return tab
