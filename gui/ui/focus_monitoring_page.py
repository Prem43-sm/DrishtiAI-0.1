from __future__ import annotations

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from features.analytics.ai_insight_engine import focus_insights
from features.analytics.analytics_database import (
    fetch_reports,
    initialize_analytics_database,
    save_focus_report,
    update_report_path,
)
from features.analytics.focus_tracking_engine import FocusTrackingEngine
from features.analytics.report_exporter import export_focus_report
from features.analytics.student_data import discover_classes, discover_students
from gui.camera_backend import open_camera_capture


class FocusCameraWorker(QThread):
    frame_ready = Signal(QImage, dict)
    stats_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, camera_id: int, student_id: str):
        super().__init__()
        self.camera_id = int(camera_id)
        self.student_id = student_id
        self.running = True
        self.engine = None

    def run(self):
        self.engine = FocusTrackingEngine()
        self.engine.reset(self.student_id)
        cap, backend_name, _ = open_camera_capture(self.camera_id)
        if cap is None:
            self.error.emit("Camera open failed.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            frame, state = self.engine.analyze_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            state["backend"] = backend_name
            self.frame_ready.emit(image, state)
            self.stats_ready.emit(self.engine.stats.to_report())
            self.msleep(30)
        cap.release()

    def final_report(self):
        if self.engine is None:
            return FocusTrackingEngine().stats.to_report()
        return self.engine.stats.to_report()

    def stop(self):
        self.running = False
        self.wait(3000)


class FocusModeMonitoringPage(QWidget):
    def __init__(self):
        super().__init__()
        initialize_analytics_database()
        self.worker = None
        self.latest_report = None
        self.latest_report_id = None
        self.report_rows = []
        self._build_ui()
        self.reload_filters()
        self.load_reports()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        title = QLabel("Focus Mode Monitoring")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#f2f2f2;")
        root.addWidget(title)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Class"))
        self.class_filter = QComboBox()
        self.class_filter.setMinimumWidth(160)
        controls.addWidget(self.class_filter)

        controls.addWidget(QLabel("Student"))
        self.student_filter = QComboBox()
        self.student_filter.setMinimumWidth(190)
        controls.addWidget(self.student_filter)

        controls.addWidget(QLabel("Camera"))
        self.camera_filter = QComboBox()
        self.camera_filter.addItems([str(i) for i in range(8)])
        controls.addWidget(self.camera_filter)

        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop and Save")
        self.refresh_btn = QPushButton("Refresh")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.refresh_btn)
        controls.addStretch()
        root.addLayout(controls)

        body = QHBoxLayout()
        self.video_label = QLabel("Camera feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 360)
        self.video_label.setStyleSheet("background:black; color:#9aa0a6;")
        body.addWidget(self.video_label, 2)

        side = QVBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size:16px; font-weight:bold;")
        side.addWidget(self.status_label)

        metric_grid = QGridLayout()
        self.metrics = {}
        for row, key in enumerate(("Focus Score", "Attention", "Distraction", "Active Time", "Inactive Time", "Movement", "Sleeping", "Looking Away")):
            label = QLabel(key)
            value = QLabel("--")
            value.setStyleSheet("font-weight:bold; color:#f8fafc;")
            metric_grid.addWidget(label, row, 0)
            metric_grid.addWidget(value, row, 1)
            self.metrics[key] = value
        side.addLayout(metric_grid)

        self.insight_label = QLabel("AI Insights: --")
        self.insight_label.setWordWrap(True)
        self.insight_label.setStyleSheet("background:#101419; border:1px solid #28313d; padding:8px;")
        side.addWidget(self.insight_label)

        export_row = QHBoxLayout()
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_excel_btn = QPushButton("Export Excel")
        export_row.addWidget(self.export_pdf_btn)
        export_row.addWidget(self.export_csv_btn)
        export_row.addWidget(self.export_excel_btn)
        side.addLayout(export_row)
        side.addStretch()
        body.addLayout(side, 1)
        root.addLayout(body, 1)

        self.report_table = QTableWidget()
        self.report_table.setColumnCount(8)
        self.report_table.setHorizontalHeaderLabels(["Student", "Date", "Focus", "Attention", "Distraction", "Movement", "Sleep", "Status"])
        self.report_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.report_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.report_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.report_table)

        self.class_filter.currentIndexChanged.connect(self._populate_students)
        self.refresh_btn.clicked.connect(self.reload_filters)
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.export_pdf_btn.clicked.connect(lambda: self.export_report("PDF"))
        self.export_csv_btn.clicked.connect(lambda: self.export_report("CSV"))
        self.export_excel_btn.clicked.connect(lambda: self.export_report("Excel"))
        self.report_table.itemSelectionChanged.connect(self.on_report_selection_changed)
        self._set_export_enabled(True)

    def reload_filters(self):
        current = self.class_filter.currentText()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("All Classes")
        self.class_filter.addItems(discover_classes())
        if current:
            index = self.class_filter.findText(current)
            if index >= 0:
                self.class_filter.setCurrentIndex(index)
        self.class_filter.blockSignals(False)
        self._populate_students()
        self.load_reports()

    def _populate_students(self):
        class_name = self.class_filter.currentText().strip() or "All Classes"
        self.student_filter.clear()
        students = discover_students(class_name)
        self.student_filter.addItems(students or ["Class Session"])

    def start_tracking(self):
        self.stop_tracking(save=False)
        student = self.student_filter.currentText().strip() or "Class Session"
        self.worker = FocusCameraWorker(int(self.camera_filter.currentText()), student)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.stats_ready.connect(self.update_stats)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Tracking")

    def stop_tracking(self, save=True):
        if self.worker is None:
            return
        worker = self.worker
        self.worker = None
        worker.stop()
        report = worker.final_report()
        worker.deleteLater()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if save:
            self.latest_report = report
            self.latest_report_id = save_focus_report(report)
            self.latest_report["id"] = self.latest_report_id
            self.update_stats(report)
            self.load_reports()
            self._set_export_enabled(True)
            self.status_label.setText(f"Status: Saved - {report['status']}")

    def update_frame(self, image, state):
        self.video_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )
        self.status_label.setText(f"Status: {state.get('status', '---')}")

    def update_stats(self, report):
        self.latest_report = report
        self.metrics["Focus Score"].setText(f"{report['focus_score']:.1f}")
        self.metrics["Attention"].setText(f"{report['attention_percentage']:.1f}%")
        self.metrics["Distraction"].setText(f"{report['distraction_percentage']:.1f}%")
        self.metrics["Active Time"].setText(f"{report['active_time']:.0f}s")
        self.metrics["Inactive Time"].setText(f"{report['inactive_time']:.0f}s")
        self.metrics["Movement"].setText(str(report["movement_count"]))
        self.metrics["Sleeping"].setText(str(report["sleep_detection_count"]))
        self.metrics["Looking Away"].setText(str(report["looking_away_count"]))
        self.insight_label.setText("AI Insights: " + " ".join(focus_insights(report)))

    def on_worker_error(self, message):
        QMessageBox.warning(self, "Focus Monitoring", message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def export_report(self, format_name):
        if self.worker is not None and self.latest_report:
            reply = QMessageBox.question(
                self,
                "Export Live Report",
                "Tracking is still running. Export the current live focus summary?",
            )
            if reply != QMessageBox.Yes:
                return

        if not self.latest_report:
            QMessageBox.information(
                self,
                "Export Report",
                "No focus report is available yet. Start tracking, then stop and save, or select a saved report from the table.",
            )
            return

        try:
            path = export_focus_report(self.latest_report, format_name)
            if self.latest_report_id:
                update_report_path("focus_reports", self.latest_report_id, path)
                self.latest_report["report_path"] = str(path)
            QMessageBox.information(self, "Export Complete", f"Report saved:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def load_reports(self):
        self.report_rows = fetch_reports("focus_reports", limit=100)
        self.report_table.setRowCount(len(self.report_rows))
        for row_index, row in enumerate(self.report_rows):
            values = [
                row.get("student_id", ""),
                row.get("date", ""),
                f"{row.get('focus_score', 0):.1f}",
                f"{row.get('attention_percentage', 0):.1f}%",
                f"{row.get('distraction_percentage', 0):.1f}%",
                str(row.get("movement_count", 0)),
                str(row.get("sleep_detection_count", 0)),
                row.get("status", ""),
            ]
            for col, value in enumerate(values):
                self.report_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        if self.report_rows and self.latest_report is None:
            self.report_table.selectRow(0)
            self.on_report_selection_changed()
        self._set_export_enabled(True)

    def on_report_selection_changed(self):
        row_index = self.report_table.currentRow()
        if row_index < 0 or row_index >= len(self.report_rows):
            return
        selected = dict(self.report_rows[row_index])
        self.latest_report = selected
        self.latest_report_id = selected.get("id")
        self.update_stats(selected)
        self._set_export_enabled(True)
        self.status_label.setText(f"Status: Selected - {selected.get('status', 'Saved Report')}")

    def _set_export_enabled(self, enabled):
        for button in (self.export_pdf_btn, self.export_csv_btn, self.export_excel_btn):
            button.setEnabled(enabled)

    def closeEvent(self, event):
        self.stop_tracking(save=False)
        super().closeEvent(event)
