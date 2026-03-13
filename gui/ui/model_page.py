import os
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model_metrics import ModelMetrics


class ModelPage(QWidget):
    def __init__(self):
        super().__init__()

        self.metrics = ModelMetrics(
            model_path="best_emotion_model.h5",
            test_dir="Final_Dataset/test"
        )

        main_layout = QVBoxLayout()

        title = QLabel("Model Performance Score")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        self.acc_label = QLabel("Accuracy: -")
        self.prec_label = QLabel("Precision: -")
        self.rec_label = QLabel("Recall: -")
        self.f1_label = QLabel("F1 Score: -")

        main_layout.addWidget(self.acc_label)
        main_layout.addWidget(self.prec_label)
        main_layout.addWidget(self.rec_label)
        main_layout.addWidget(self.f1_label)

        self.eval_btn = QPushButton("Evaluate Model")
        self.eval_btn.clicked.connect(self.evaluate_model)
        main_layout.addWidget(self.eval_btn)

        chart_layout = QHBoxLayout()

        self.cm_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.pie_canvas = FigureCanvas(Figure(figsize=(4, 3)))

        chart_layout.addWidget(self.cm_canvas)
        chart_layout.addWidget(self.pie_canvas)

        main_layout.addLayout(chart_layout)

        self.report_table = QTableWidget()
        self.report_table.setColumnCount(5)
        self.report_table.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1-Score", "Support"]
        )

        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.report_table.setStyleSheet("""
        QTableWidget {
            background-color: #111;
            color: white;
            gridline-color: #444;
        }
        QHeaderView::section {
            background-color: #222;
            color: white;
            font-weight: bold;
        }
        """)

        main_layout.addWidget(self.report_table)

        self.setLayout(main_layout)

    def _apply_dark_chart_theme(self, fig, ax):
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#1b1b1b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#666666")

    def evaluate_model(self):

        acc, prec, rec, f1, cm, report = self.metrics.evaluate()

        self.acc_label.setText(f"Accuracy: {acc:.4f}")
        self.prec_label.setText(f"Precision: {prec:.4f}")
        self.rec_label.setText(f"Recall: {rec:.4f}")
        self.f1_label.setText(f"F1 Score: {f1:.4f}")

        self.plot_confusion_matrix(cm)
        self.plot_pie_chart(cm)
        self.populate_report_table(report)

    def plot_confusion_matrix(self, cm):

        fig = self.cm_canvas.figure
        fig.clear()

        ax = fig.add_subplot(111)
        self._apply_dark_chart_theme(fig, ax)
        im = ax.imshow(cm)

        ax.set_title("Confusion Matrix")
        cbar = fig.colorbar(im)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("#666666")
        for tick in cbar.ax.get_yticklabels():
            tick.set_color("white")

        self.cm_canvas.draw()

    def plot_pie_chart(self, cm):

        fig = self.pie_canvas.figure
        fig.clear()

        ax = fig.add_subplot(111)
        self._apply_dark_chart_theme(fig, ax)

        class_totals = cm.sum(axis=1)

        labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

        _, text_labels, text_autopct = ax.pie(
            class_totals,
            labels=labels,
            autopct="%1.1f%%",
            textprops={"color": "white"},
        )
        for text in text_labels + text_autopct:
            text.set_color("white")
        ax.set_title("Class Distribution")

        self.pie_canvas.draw()

    def populate_report_table(self, report_dict):

        classes = [
            c for c in report_dict.keys()
            if c not in ("accuracy", "macro avg", "weighted avg")
        ]

        self.report_table.setRowCount(len(classes) + 3)

        row = 0

        for cls in classes:
            data = report_dict[cls]

            self.report_table.setItem(row, 0, QTableWidgetItem(cls))
            self.report_table.setItem(row, 1, QTableWidgetItem(f"{data['precision']:.2f}"))
            self.report_table.setItem(row, 2, QTableWidgetItem(f"{data['recall']:.2f}"))
            self.report_table.setItem(row, 3, QTableWidgetItem(f"{data['f1-score']:.2f}"))
            self.report_table.setItem(row, 4, QTableWidgetItem(str(int(data['support']))))

            row += 1

        acc = report_dict["accuracy"]

        self.report_table.setItem(row, 0, QTableWidgetItem("Accuracy"))
        self.report_table.setItem(row, 3, QTableWidgetItem(f"{acc:.2f}"))

        row += 1

        for key in ["macro avg", "weighted avg"]:

            data = report_dict[key]

            self.report_table.setItem(row, 0, QTableWidgetItem(key))
            self.report_table.setItem(row, 1, QTableWidgetItem(f"{data['precision']:.2f}"))
            self.report_table.setItem(row, 2, QTableWidgetItem(f"{data['recall']:.2f}"))
            self.report_table.setItem(row, 3, QTableWidgetItem(f"{data['f1-score']:.2f}"))
            self.report_table.setItem(row, 4, QTableWidgetItem(str(int(data['support']))))

            row += 1
