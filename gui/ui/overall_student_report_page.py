from __future__ import annotations

from datetime import datetime

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from features.analytics.overall_student_report_engine import OverallStudentReportEngine
from features.analytics.report_exporter import export_overall_student_report
from features.analytics.student_data import discover_classes, discover_students, load_emotion_samples


class OverallStudentReportPage(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = OverallStudentReportEngine()
        self.samples = pd.DataFrame()
        self.class_reports = []
        self.current_report = None
        self._build_ui()
        self.reload_filters()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        title = QLabel("Overall Student Report")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#f2f2f2;")
        root.addWidget(title)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Class"))
        self.class_filter = QComboBox()
        self.class_filter.setMinimumWidth(160)
        filters.addWidget(self.class_filter)

        filters.addWidget(QLabel("Student"))
        self.student_filter = QComboBox()
        self.student_filter.setMinimumWidth(190)
        filters.addWidget(self.student_filter)

        today = datetime.now()
        filters.addWidget(QLabel("Month"))
        self.month_filter = QSpinBox()
        self.month_filter.setRange(1, 12)
        self.month_filter.setValue(today.month)
        filters.addWidget(self.month_filter)

        filters.addWidget(QLabel("Year"))
        self.year_filter = QSpinBox()
        self.year_filter.setRange(2020, 2100)
        self.year_filter.setValue(today.year)
        filters.addWidget(self.year_filter)

        self.generate_btn = QPushButton("Generate Overall Report")
        self.refresh_btn = QPushButton("Refresh Data")
        filters.addWidget(self.generate_btn)
        filters.addWidget(self.refresh_btn)
        filters.addStretch()
        root.addLayout(filters)

        score_grid = QGridLayout()
        self.metric_cards = {}
        for index, key in enumerate(("Overall Points", "Class Rank", "Attendance", "Emotion", "Performance", "Focus")):
            card = QFrame()
            card.setStyleSheet("QFrame { background:#171b22; border:1px solid #2c3440; border-radius:6px; padding:8px; }")
            card_layout = QVBoxLayout(card)
            label = QLabel(key)
            label.setStyleSheet("color:#aeb8c7; font-size:12px;")
            value = QLabel("--")
            value.setStyleSheet("font-size:20px; font-weight:bold; color:#f5f7fb;")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            self.metric_cards[key] = value
            score_grid.addWidget(card, index // 3, index % 3)
        root.addLayout(score_grid)

        content = QHBoxLayout()

        left = QVBoxLayout()
        self.fig = Figure(figsize=(5.0, 3.0), tight_layout=True)
        self.fig.patch.set_facecolor("#0b0f14")
        self.canvas = FigureCanvas(self.fig)
        left.addWidget(self.canvas, 1)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(170)
        self.report_text.setStyleSheet("background:#101419; border:1px solid #28313d;")
        left.addWidget(self.report_text)
        content.addLayout(left, 1)

        self.rank_table = QTableWidget()
        self.rank_table.setColumnCount(8)
        self.rank_table.setHorizontalHeaderLabels(
            ["Rank", "Student", "Overall", "Attendance", "Emotion", "Performance", "Focus", "Status"]
        )
        self.rank_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rank_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rank_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content.addWidget(self.rank_table, 1)
        root.addLayout(content, 1)

        exports = QHBoxLayout()
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_excel_btn = QPushButton("Export Excel")
        exports.addStretch()
        exports.addWidget(self.export_pdf_btn)
        exports.addWidget(self.export_csv_btn)
        exports.addWidget(self.export_excel_btn)
        root.addLayout(exports)

        self.class_filter.currentIndexChanged.connect(self._populate_students)
        self.refresh_btn.clicked.connect(self.reload_filters)
        self.generate_btn.clicked.connect(self.generate_report)
        self.export_pdf_btn.clicked.connect(lambda: self.export_report("PDF"))
        self.export_csv_btn.clicked.connect(lambda: self.export_report("CSV"))
        self.export_excel_btn.clicked.connect(lambda: self.export_report("Excel"))
        self.rank_table.itemSelectionChanged.connect(self.on_rank_selection_changed)
        self._set_export_enabled(False)
        self._draw_empty_chart()

    def reload_filters(self):
        self.samples = load_emotion_samples()
        current_class = self.class_filter.currentText()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItems(discover_classes())
        if current_class:
            index = self.class_filter.findText(current_class)
            if index >= 0:
                self.class_filter.setCurrentIndex(index)
        self.class_filter.blockSignals(False)
        self._populate_students()

    def _populate_students(self):
        class_name = self.class_filter.currentText().strip()
        self.student_filter.clear()
        self.student_filter.addItems(discover_students(class_name))

    def generate_report(self):
        class_name = self.class_filter.currentText().strip()
        student = self.student_filter.currentText().strip()
        if not class_name:
            QMessageBox.warning(self, "Overall Report", "Select a class first.")
            return
        if not student:
            QMessageBox.warning(self, "Overall Report", "Select a student first.")
            return

        self.class_reports = self.engine.build_class_report(
            self.samples,
            class_name,
            self.month_filter.value(),
            self.year_filter.value(),
        )
        self._render_rank_table()
        selected = next((row for row in self.class_reports if row["student_id"] == student), None)
        if selected is None:
            QMessageBox.information(self, "Overall Report", "No report data found for the selected student.")
            return
        self.current_report = selected
        self._show_report(selected)
        self._select_student_row(student)
        self._set_export_enabled(True)

    def _render_rank_table(self):
        self.rank_table.setRowCount(len(self.class_reports))
        for row_index, row in enumerate(self.class_reports):
            values = [
                row.get("rank", ""),
                row.get("student_id", ""),
                f"{row.get('overall_points', 0):.1f}",
                f"{row.get('attendance_score', 0):.1f}",
                f"{row.get('emotion_score', 0):.1f}",
                f"{row.get('performance_score', 0):.1f}",
                self._format_score(row.get("focus_score")),
                row.get("overall_status", ""),
            ]
            for col, value in enumerate(values):
                self.rank_table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def _select_student_row(self, student_id: str):
        for index, row in enumerate(self.class_reports):
            if row.get("student_id") == student_id:
                self.rank_table.selectRow(index)
                break

    def on_rank_selection_changed(self):
        row_index = self.rank_table.currentRow()
        if row_index < 0 or row_index >= len(self.class_reports):
            return
        self.current_report = self.class_reports[row_index]
        self.student_filter.setCurrentText(self.current_report["student_id"])
        self._show_report(self.current_report)
        self._set_export_enabled(True)

    def _show_report(self, report):
        total = len(self.class_reports)
        self.metric_cards["Overall Points"].setText(f"{report['overall_points']:.1f}/100")
        self.metric_cards["Class Rank"].setText(f"#{report['rank']} of {total}")
        self.metric_cards["Attendance"].setText(
            f"{report['attendance_score']:.1f}% ({report['present_sessions']}/{report['total_sessions']})"
        )
        self.metric_cards["Emotion"].setText(f"{report['emotion_score']:.1f}")
        self.metric_cards["Performance"].setText(f"{report['performance_score']:.1f}")
        if report.get("has_focus_data"):
            self.metric_cards["Focus"].setText(f"{report['focus_score']:.1f}")
        else:
            self.metric_cards["Focus"].setText("No Data")
        self.report_text.setPlainText(
            f"Student: {report['student_id']}\n"
            f"Class: {report['class']}\n"
            f"Month/Year: {report['month']}/{report['year']}\n"
            f"Overall Status: {report['overall_status']}\n\n"
            f"Summary:\n{report['summary']}\n\n"
            f"Recommendations:\n" + "\n".join(f"- {item}" for item in report["recommendations"])
        )
        self._draw_chart(report)

    def _draw_empty_chart(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f141b")
        ax.text(0.5, 0.5, "Generate overall report", ha="center", va="center", color="#cbd5e1")
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _draw_chart(self, report):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f141b")
        labels = ["Attendance", "Emotion", "Performance", "Focus", "Overall"]
        values = [
            report["attendance_score"],
            report["emotion_score"],
            report["performance_score"],
            report["focus_score"] if report.get("has_focus_data") else 0,
            report["overall_points"],
        ]
        ax.bar(labels, values, color=["#2563eb", "#22c55e", "#f59e0b", "#38bdf8", "#a855f7"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Points", color="#e6ecf5")
        ax.set_title("Score Breakdown", color="#f8fafc")
        ax.tick_params(colors="#d7dde8")
        for spine in ax.spines.values():
            spine.set_color("#425167")
        self.canvas.draw_idle()

    def _format_score(self, value):
        if value is None:
            return "No Data"
        return f"{float(value):.1f}"

    def export_report(self, format_name):
        if not self.current_report:
            return
        try:
            path = export_overall_student_report(self.current_report, format_name)
            QMessageBox.information(self, "Export Complete", f"Report saved:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _set_export_enabled(self, enabled):
        for button in (self.export_pdf_btn, self.export_csv_btn, self.export_excel_btn):
            button.setEnabled(enabled)
