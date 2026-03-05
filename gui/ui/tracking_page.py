from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import QTimer
from features.tracking.live_tracker import get_all_locations, get_location


class TrackingPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        title = QLabel("Live Tracking")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # 🔍 SEARCH BAR
        search_layout = QHBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter Class")

        self.search_btn = QPushButton("Track")

        search_layout.addWidget(self.name_input)
        search_layout.addWidget(self.class_input)
        search_layout.addWidget(self.search_btn)

        layout.addLayout(search_layout)

        # 🧾 RESULT LABEL
        self.result = QLabel("Status: ---")
        layout.addWidget(self.result)

        # 📋 LIVE TABLE
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Status", "Camera", "Class", "Last Seen"]
        )
        layout.addWidget(self.table)

        self.setLayout(layout)

        # ⏱ TIMER
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_live_data)
        self.timer.start(1000)

        self.search_btn.clicked.connect(self.search_student)

    # ---------------- SEARCH MODE ----------------
    def search_student(self):

        name = self.name_input.text()
        data = get_location(name)

        if data:
            self.result.setText(
                f"🟢 LIVE\nCamera: {data['camera']} | Class: {data['class']} | {data['time']}"
            )
        else:
            self.result.setText("🔴 Not Found")

    # ---------------- LIVE TABLE ----------------
    def load_live_data(self):

        data = get_all_locations()
        self.table.setRowCount(len(data))

        for row, student in enumerate(data):

            status = "LIVE ✅" if student["live"] else "OFFLINE ❌"

            self.table.setItem(row, 0, QTableWidgetItem(student["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(status))
            self.table.setItem(row, 2, QTableWidgetItem(student["camera"]))
            self.table.setItem(row, 3, QTableWidgetItem(student["class"]))
            self.table.setItem(row, 4, QTableWidgetItem(student["time"]))
