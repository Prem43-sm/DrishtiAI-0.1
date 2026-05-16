from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap, QDesktopServices, QCursor
from PySide6.QtCore import Qt, QUrl, QTime, QThread, Signal
import os
import json
import shutil
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
        self.tabs.addTab(self.face_db_tab(), "Face Database")
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

        base = str(KNOWN_FACES_DIR)
        if not os.path.exists(base):
            return

        for item in sorted(os.listdir(base)):
            full = os.path.join(base, item)
            if os.path.isdir(full):
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
            self.student_list.clear()
            self.student_count.setText("Total Students: 0")

        self.load_classes()

    def load_students(self, item):
        self.current_class = item.text()
        self.student_list.clear()

        path = os.path.join(str(KNOWN_FACES_DIR), self.current_class)
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

        np.save(os.path.join(str(KNOWN_FACES_DIR), self.current_class, f"{name}.npy"), enc)

        self.student_name.clear()
        self.load_students(QListWidgetItem(self.current_class))

        # 🔥 IMPORTANT FIX
        FaceMemory.get_instance().reload()

        QMessageBox.information(self, "Success", "Student Added ✅")

    def delete_student(self):
        item = self.student_list.currentItem()
        if not item:
            return

        path = os.path.join(str(KNOWN_FACES_DIR), self.current_class, f"{item.text()}.npy")

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
