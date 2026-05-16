from __future__ import annotations

from datetime import datetime

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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

from features.analytics.ai_insight_engine import emotion_insights
from features.analytics.analytics_database import (
    initialize_analytics_database,
    save_emotion_report,
    update_report_path,
)
from features.analytics.emotion_performance_engine import EmotionPerformanceEngine
from features.analytics.report_exporter import export_emotion_report
from features.analytics.student_data import EMOTION_LABELS, discover_classes, discover_students, load_emotion_samples


class EmotionPerformanceAnalyticsPage(QWidget):
    def __init__(self):
        super().__init__()
        initialize_analytics_database()
        self.engine = EmotionPerformanceEngine()
        self.samples = pd.DataFrame()
        self.current_report = None
        self.current_report_id = None
        self._build_ui()
        self.reload_filters()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        title = QLabel("Emotion Performance Analytics")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#f2f2f2;")
        root.addWidget(title)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Class"))
        self.class_filter = QComboBox()
        self.class_filter.setMinimumWidth(160)
        filters.addWidget(self.class_filter)

        filters.addWidget(QLabel("Section"))
        self.section_filter = QComboBox()
        self.section_filter.addItem("All Sections")
        filters.addWidget(self.section_filter)

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

        self.analyze_btn = QPushButton("Generate Analysis")
        self.refresh_btn = QPushButton("Refresh Data")
        filters.addWidget(self.analyze_btn)
        filters.addWidget(self.refresh_btn)
        filters.addStretch()
        root.addLayout(filters)

        score_grid = QGridLayout()
        self.metric_cards = {}
        for index, key in enumerate(("Emotion Score", "Engagement Score", "Stability Score", "Performance")):
            card = QFrame()
            card.setStyleSheet("QFrame { background:#171b22; border:1px solid #2c3440; border-radius:6px; padding:8px; }")
            card_layout = QVBoxLayout(card)
            label = QLabel(key)
            label.setStyleSheet("color:#aeb8c7; font-size:12px;")
            value = QLabel("--")
            value.setStyleSheet("font-size:22px; font-weight:bold; color:#f5f7fb;")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            self.metric_cards[key] = value
            score_grid.addWidget(card, 0, index)
        root.addLayout(score_grid)

        chart_row = QHBoxLayout()
        self.pie_fig, self.pie_canvas = self._new_chart()
        self.trend_fig, self.trend_canvas = self._new_chart()
        self.focus_fig, self.focus_canvas = self._new_chart()
        self.score_fig, self.score_canvas = self._new_chart()
        chart_row.addWidget(self.pie_canvas, 1)
        chart_row.addWidget(self.trend_canvas, 1)
        chart_row.addWidget(self.focus_canvas, 1)
        chart_row.addWidget(self.score_canvas, 1)
        root.addLayout(chart_row, 1)

        lower = QHBoxLayout()
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(160)
        self.report_text.setStyleSheet("background:#101419; border:1px solid #28313d;")
        lower.addWidget(self.report_text, 2)

        self.percent_table = QTableWidget()
        self.percent_table.setColumnCount(2)
        self.percent_table.setHorizontalHeaderLabels(["Emotion", "Percentage"])
        lower.addWidget(self.percent_table, 1)
        root.addLayout(lower)

        exports = QHBoxLayout()
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_excel_btn = QPushButton("Export Excel")
        exports.addStretch()
        exports.addWidget(self.export_pdf_btn)
        exports.addWidget(self.export_csv_btn)
        exports.addWidget(self.export_excel_btn)
        root.addLayout(exports)

        self.class_filter.currentIndexChanged.connect(self._on_class_change)
        self.refresh_btn.clicked.connect(self.reload_filters)
        self.analyze_btn.clicked.connect(self.generate_report)
        self.export_pdf_btn.clicked.connect(lambda: self.export_report("PDF"))
        self.export_csv_btn.clicked.connect(lambda: self.export_report("CSV"))
        self.export_excel_btn.clicked.connect(lambda: self.export_report("Excel"))
        self._set_export_enabled(False)
        self._draw_empty_charts("Generate analysis")

    def _new_chart(self):
        fig = Figure(figsize=(3.3, 2.6), tight_layout=True)
        fig.patch.set_facecolor("#0b0f14")
        canvas = FigureCanvas(fig)
        return fig, canvas

    def reload_filters(self):
        self.samples = load_emotion_samples()
        current_class = self.class_filter.currentText()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("All Classes")
        self.class_filter.addItems(discover_classes())
        if current_class:
            idx = self.class_filter.findText(current_class)
            if idx >= 0:
                self.class_filter.setCurrentIndex(idx)
        self.class_filter.blockSignals(False)
        self._populate_sections()
        self._populate_students()

    def _on_class_change(self):
        self._populate_sections()
        self._populate_students()

    def _populate_sections(self):
        selected_class = self.class_filter.currentText().strip()
        self.section_filter.clear()
        self.section_filter.addItem("All Sections")
        if self.samples.empty or "section" not in self.samples.columns:
            return
        data = self.samples
        if selected_class and selected_class != "All Classes":
            data = data[data["class"].astype(str) == selected_class]
        sections = sorted(s for s in data["section"].dropna().astype(str).unique().tolist() if s and s.lower() != "nan")
        self.section_filter.addItems(sections)

    def _populate_students(self):
        class_name = self.class_filter.currentText().strip() or "All Classes"
        self.student_filter.clear()
        self.student_filter.addItems(discover_students(class_name))

    def _filtered_samples(self):
        data = self.samples.copy()
        class_name = self.class_filter.currentText().strip()
        section = self.section_filter.currentText().strip()
        if not data.empty and class_name and class_name != "All Classes":
            data = data[data["class"].astype(str) == class_name]
        if not data.empty and section and section != "All Sections" and "section" in data.columns:
            data = data[data["section"].astype(str) == section]
        return data

    def generate_report(self):
        student = self.student_filter.currentText().strip()
        if not student:
            QMessageBox.warning(self, "Analytics", "Select a student first.")
            return
        data = self._filtered_samples()
        report = self.engine.build_monthly_report(
            data,
            student,
            self.month_filter.value(),
            self.year_filter.value(),
        )
        self.current_report = report
        self.current_report_id = save_emotion_report(report)
        self._update_metrics(report)
        self._draw_charts(report)
        self._update_table(report)
        self._update_report_text(report)
        self._set_export_enabled(True)

    def export_report(self, format_name):
        if not self.current_report:
            return
        try:
            path = export_emotion_report(self.current_report, format_name)
            if self.current_report_id:
                update_report_path("emotion_reports", self.current_report_id, path)
            QMessageBox.information(self, "Export Complete", f"Report saved:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _set_export_enabled(self, enabled):
        for button in (self.export_pdf_btn, self.export_csv_btn, self.export_excel_btn):
            button.setEnabled(enabled)

    def _update_metrics(self, report):
        self.metric_cards["Emotion Score"].setText(f"{report['emotion_score']:.1f}")
        self.metric_cards["Engagement Score"].setText(f"{report['engagement_score']:.1f}")
        self.metric_cards["Stability Score"].setText(f"{report['stability_score']:.1f}")
        self.metric_cards["Performance"].setText(f"{report['performance_score']:.1f} {report['performance_status']}")

    def _update_table(self, report):
        self.percent_table.setRowCount(len(EMOTION_LABELS))
        for row, label in enumerate(EMOTION_LABELS):
            self.percent_table.setItem(row, 0, QTableWidgetItem(label.title()))
            self.percent_table.setItem(row, 1, QTableWidgetItem(f"{report[f'{label}_percentage']:.2f}%"))

    def _update_report_text(self, report):
        recommendations = "\n".join(f"- {item}" for item in report["recommendations"])
        insights = "\n".join(f"- {item}" for item in emotion_insights(report))
        self.report_text.setPlainText(
            f"Student: {report['student_id']}\n"
            f"Month/Year: {report['month']}/{report['year']}\n"
            f"Samples: {report['total_samples']}\n"
            f"Positive vs Negative Ratio: {report['positive_negative_ratio']}\n\n"
            f"Emotional Summary:\n{report['summary']}\n\n"
            f"AI Insights:\n{insights}\n\n"
            f"Recommendations:\n{recommendations}"
        )

    def _draw_empty_charts(self, message):
        for fig, canvas, title in (
            (self.pie_fig, self.pie_canvas, "Emotion Distribution"),
            (self.trend_fig, self.trend_canvas, "Monthly Trend"),
            (self.focus_fig, self.focus_canvas, "Focus vs Emotion"),
            (self.score_fig, self.score_canvas, "Performance Score"),
        ):
            fig.clear()
            ax = fig.add_subplot(111)
            self._style_axis(ax)
            ax.text(0.5, 0.5, message, ha="center", va="center", color="#cbd5e1")
            ax.set_title(title)
            ax.set_axis_off()
            canvas.draw_idle()

    def _draw_charts(self, report):
        self._draw_pie(report)
        self._draw_trend(report)
        self._draw_focus_emotion(report)
        self._draw_score(report)

    def _draw_pie(self, report):
        self.pie_fig.clear()
        ax = self.pie_fig.add_subplot(111)
        values = [report["counts"][label] for label in EMOTION_LABELS]
        if sum(values) == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", color="#cbd5e1")
            ax.set_axis_off()
        else:
            ax.pie(values, labels=[l.title() for l in EMOTION_LABELS], autopct="%1.0f%%", startangle=140, textprops={"color": "#e5e7eb", "fontsize": 8})
        ax.set_title("Emotion Distribution", color="#f8fafc")
        self.pie_canvas.draw_idle()

    def _draw_trend(self, report):
        self.trend_fig.clear()
        ax = self.trend_fig.add_subplot(111)
        self._style_axis(ax)
        trend = report["trend"]
        if trend.empty:
            ax.text(0.5, 0.5, "No monthly trend", ha="center", va="center", color="#cbd5e1")
        else:
            ax.plot(trend["date"].dt.day, trend["score"], color="#4ea1ff", marker="o", linewidth=1.8)
            ax.set_xlabel("Day")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 100)
            ax.grid(color="#273240", alpha=0.35)
        ax.set_title("Monthly Trend", color="#f8fafc")
        self.trend_canvas.draw_idle()

    def _draw_focus_emotion(self, report):
        self.focus_fig.clear()
        ax = self.focus_fig.add_subplot(111)
        self._style_axis(ax)
        ax.bar(["Engagement", "Stability"], [report["engagement_score"], report["stability_score"]], color=["#22c55e", "#f59e0b"])
        ax.set_ylim(0, 100)
        ax.set_title("Focus vs Emotion", color="#f8fafc")
        self.focus_canvas.draw_idle()

    def _draw_score(self, report):
        self.score_fig.clear()
        ax = self.score_fig.add_subplot(111)
        self._style_axis(ax)
        ax.barh(["Performance"], [report["performance_score"]], color="#38bdf8")
        ax.set_xlim(0, 100)
        ax.set_title(report["performance_status"], color="#f8fafc")
        self.score_canvas.draw_idle()

    def _style_axis(self, ax):
        ax.set_facecolor("#0f141b")
        for spine in ax.spines.values():
            spine.set_color("#425167")
        ax.tick_params(colors="#d7dde8")
        ax.xaxis.label.set_color("#e6ecf5")
        ax.yaxis.label.set_color("#e6ecf5")

