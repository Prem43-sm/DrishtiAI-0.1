import json
import os
import shutil

from PySide6.QtCore import QTime, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLineEdit,
    QPushButton,
    QLabel,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSpinBox,
    QTimeEdit,
    QMessageBox,
)


TIMETABLE_DIR = "timetable"
CLASS_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"


class TimeTablePage(QWidget):
    def __init__(self):
        super().__init__()

        os.makedirs(TIMETABLE_DIR, exist_ok=True)
        os.makedirs(CLASS_DIR, exist_ok=True)
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)

        main_outer = QVBoxLayout(self)

        title = QLabel("Time-Table Editor")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_outer.addWidget(title)

        main = QHBoxLayout()

        left = QVBoxLayout()
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.load_timetable)

        self.new_class = QLineEdit()
        self.new_class.setPlaceholderText("New Class")

        add_class = QPushButton("Add Class")
        add_class.clicked.connect(self.add_class)

        delete_class = QPushButton("Delete Class")
        delete_class.clicked.connect(self.delete_class)

        left.addWidget(QLabel("CLASSES"))
        left.addWidget(self.class_list)
        left.addWidget(self.new_class)
        left.addWidget(add_class)
        left.addWidget(delete_class)

        center = QVBoxLayout()
        self.day_selector = QComboBox()
        self.day_selector.addItems(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        )
        self.day_selector.currentTextChanged.connect(self.load_day)

        self.period_table = QTableWidget(0, 4)
        self.period_table.setHorizontalHeaderLabels(["Period", "Start", "End", "Subject"])
        self.period_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.period_table.cellClicked.connect(self.fill_editor_from_table)

        hint = QLabel("Select a row to edit -> modify -> click Add / Update")
        hint.setStyleSheet("color:gray; font-size:11px;")

        center.addWidget(QLabel("DAY"))
        center.addWidget(self.day_selector)
        center.addWidget(self.period_table)
        center.addWidget(hint)

        right = QVBoxLayout()
        right.addWidget(QLabel("PERIOD EDITOR"))

        self.period_no = QSpinBox()
        self.period_no.setMinimum(1)
        self.period_no.setPrefix("Period ")

        self.start_time = QTimeEdit()
        self.start_time.setDisplayFormat("HH:mm")

        self.end_time = QTimeEdit()
        self.end_time.setDisplayFormat("HH:mm")

        self.subject = QLineEdit()
        self.subject.setPlaceholderText("Subject name")

        add_period = QPushButton("Add / Update Period")
        add_period.clicked.connect(self.save_period)

        right.addWidget(QLabel("Period No"))
        right.addWidget(self.period_no)
        right.addWidget(QLabel("Start Time"))
        right.addWidget(self.start_time)
        right.addWidget(QLabel("End Time"))
        right.addWidget(self.end_time)
        right.addWidget(QLabel("Subject"))
        right.addWidget(self.subject)
        right.addSpacing(10)
        right.addWidget(add_period)
        right.addStretch()

        main.addLayout(left, 1)
        main.addLayout(center, 2)
        main.addLayout(right, 1)
        main_outer.addLayout(main)

        self.load_classes()

    def load_classes(self):
        current = self.class_list.currentItem().text() if self.class_list.currentItem() else ""
        self.class_list.clear()
        classes = []

        if not os.path.exists(CLASS_DIR):
            return

        for cls in sorted(os.listdir(CLASS_DIR)):
            full = os.path.join(CLASS_DIR, cls)
            if not os.path.isdir(full):
                continue
            classes.append(cls)
            self.class_list.addItem(cls)
            self._ensure_timetable_file(cls)

        if current and current in classes:
            hits = self.class_list.findItems(current, Qt.MatchExactly)
            if hits:
                self.class_list.setCurrentItem(hits[0])

    def _ensure_timetable_file(self, cls):
        path = os.path.join(TIMETABLE_DIR, f"{cls}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"days": {}}, f, indent=4)

    def add_class(self):
        name = self.new_class.text().strip()
        if not name:
            return

        os.makedirs(os.path.join(CLASS_DIR, name), exist_ok=True)
        os.makedirs(os.path.join(ATTENDANCE_DIR, name), exist_ok=True)
        self._ensure_timetable_file(name)

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

        face_path = os.path.join(CLASS_DIR, cls)
        attendance_path = os.path.join(ATTENDANCE_DIR, cls)
        tt_path = os.path.join(TIMETABLE_DIR, f"{cls}.json")

        if os.path.exists(face_path):
            shutil.rmtree(face_path, ignore_errors=True)
        if os.path.exists(attendance_path):
            shutil.rmtree(attendance_path, ignore_errors=True)
        if os.path.exists(tt_path):
            os.remove(tt_path)

        if hasattr(self, "current_class") and self.current_class == cls:
            delattr(self, "current_class")
            self.period_table.setRowCount(0)

        self.load_classes()

    def load_timetable(self):
        item = self.class_list.currentItem()
        if not item:
            return
        self.current_class = item.text()
        self._ensure_timetable_file(self.current_class)
        self.load_day()

    def load_day(self):
        if not hasattr(self, "current_class"):
            return

        self.period_table.setRowCount(0)
        path = os.path.join(TIMETABLE_DIR, f"{self.current_class}.json")
        self._ensure_timetable_file(self.current_class)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        day = self.day_selector.currentText()
        periods = sorted(data.get("days", {}).get(day, []), key=lambda x: x.get("period", 0))

        for row, p in enumerate(periods):
            self.period_table.insertRow(row)
            self.period_table.setItem(row, 0, QTableWidgetItem(str(p.get("period", ""))))
            self.period_table.setItem(row, 1, QTableWidgetItem(str(p.get("start", ""))))
            self.period_table.setItem(row, 2, QTableWidgetItem(str(p.get("end", ""))))
            self.period_table.setItem(row, 3, QTableWidgetItem(str(p.get("subject", ""))))

    def fill_editor_from_table(self, row, column):
        _ = column
        self.period_no.setValue(int(self.period_table.item(row, 0).text()))
        self.start_time.setTime(QTime.fromString(self.period_table.item(row, 1).text(), "HH:mm"))
        self.end_time.setTime(QTime.fromString(self.period_table.item(row, 2).text(), "HH:mm"))
        self.subject.setText(self.period_table.item(row, 3).text())

    def save_period(self):
        if not hasattr(self, "current_class"):
            return

        path = os.path.join(TIMETABLE_DIR, f"{self.current_class}.json")
        self._ensure_timetable_file(self.current_class)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        day = self.day_selector.currentText()
        data.setdefault("days", {})
        data["days"].setdefault(day, [])

        period_data = {
            "period": self.period_no.value(),
            "start": self.start_time.time().toString("HH:mm"),
            "end": self.end_time.time().toString("HH:mm"),
            "subject": self.subject.text(),
        }

        updated = False
        for i, p in enumerate(data["days"][day]):
            if p.get("period") == period_data["period"]:
                data["days"][day][i] = period_data
                updated = True
                break

        if not updated:
            data["days"][day].append(period_data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        self.clear_editor()
        self.load_day()

    def clear_editor(self):
        self.period_no.setValue(1)
        self.start_time.setTime(QTime(0, 0))
        self.end_time.setTime(QTime(0, 0))
        self.subject.clear()

    def showEvent(self, event):
        self.load_classes()
        super().showEvent(event)
