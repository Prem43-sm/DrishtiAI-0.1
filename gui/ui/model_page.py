import csv
import json
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.project_paths import MODELS_DIR


class ScaledImageLabel(QLabel):
    def __init__(self, image_path: Path, minimum_height: int = 280):
        super().__init__()
        self._image_path = Path(image_path)
        self._original_pixmap = QPixmap(str(self._image_path))
        self._base_minimum_height = minimum_height

        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(minimum_height)
        self.setStyleSheet(
            "QLabel {"
            " background-color: #161616;"
            " border: 1px solid #2e2e2e;"
            " border-radius: 12px;"
            " padding: 10px;"
            "}"
        )

        if self._original_pixmap.isNull():
            self.setText(f"Preview not available\n{self._image_path.name}")
        else:
            self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def hasHeightForWidth(self):
        return not self._original_pixmap.isNull()

    def heightForWidth(self, width):
        if self._original_pixmap.isNull():
            return self._base_minimum_height

        available_width = max(240, width - 24)
        scaled_height = int(
            self._original_pixmap.height() * available_width
            / max(1, self._original_pixmap.width())
        )
        return max(self._base_minimum_height, scaled_height + 24)

    def sizeHint(self):
        if self._original_pixmap.isNull():
            return super().sizeHint()

        width = self.width() if self.width() > 0 else min(
            self._original_pixmap.width() + 24, 1200
        )
        return QSize(width, self.heightForWidth(width))

    def _refresh_pixmap(self):
        if self._original_pixmap.isNull():
            return

        target_width = max(240, self.width() - 24)
        scaled = self._original_pixmap.scaledToWidth(
            target_width, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        target_height = max(self._base_minimum_height, scaled.height() + 24)
        if self.height() != target_height:
            self.setFixedHeight(target_height)
        self.updateGeometry()


class ModelPage(QWidget):
    def __init__(self):
        super().__init__()

        self.performance_dir = MODELS_DIR / "Model_performance"
        self.summary = {}
        self.class_rows = []
        self.confusion_rows = []
        self.report_text = ""

        self._build_ui()
        self.reload_performance_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()

        title_block = QVBoxLayout()
        title = QLabel("Model Performance")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        subtitle = QLabel("Saved evaluation dashboard from models/Model_performance")
        subtitle.setStyleSheet("color: #a9a9a9;")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        header.addLayout(title_block)
        header.addStretch()

        self.reload_btn = QPushButton("Reload Performance Data")
        self.reload_btn.clicked.connect(self.reload_performance_data)
        header.addWidget(self.reload_btn)

        layout.addLayout(header)

        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #8f8f8f;")
        layout.addWidget(self.path_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #d0d0d0;")
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

    def reload_performance_data(self):
        self.summary = self._load_json("metrics_summary.json")
        self.class_rows = self._load_csv("class_summary.csv")
        self.confusion_rows = self._load_csv("top_confusions.csv")
        self.report_text = self._load_text("report.txt")

        self.path_label.setText(f"Source Folder: {self.performance_dir}")
        self.status_label.setText(self._build_status_text())
        self._populate_tabs()

    def _build_status_text(self):
        if not self.performance_dir.exists():
            return "Performance folder not found. Add the evaluation files to models/Model_performance."

        evaluated_at = self.summary.get("evaluated_at", "--")
        images = self.summary.get("test_images", "--")
        classes = self.summary.get("num_classes", "--")
        return (
            f"Loaded saved evaluation data. Evaluated at: {evaluated_at} | "
            f"Test images: {images} | Classes: {classes}"
        )

    def _populate_tabs(self):
        self.tabs.clear()
        self.tabs.addTab(self._build_overview_tab(), "Overview")
        self.tabs.addTab(self._build_class_scores_tab(), "Class Scores")
        self.tabs.addTab(self._build_confusions_tab(), "Confusions")
        self.tabs.addTab(self._build_distributions_tab(), "Distributions")
        self.tabs.addTab(self._build_samples_tab(), "Samples")
        self.tabs.addTab(self._build_reports_tab(), "Reports")

    def _build_overview_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        cards = QWidget()
        card_grid = QGridLayout(cards)
        card_grid.setContentsMargins(0, 0, 0, 0)
        card_grid.setHorizontalSpacing(10)
        card_grid.setVerticalSpacing(10)

        metric_cards = [
            ("Accuracy", self._format_percent(self.summary.get("accuracy"))),
            (
                "Weighted Precision",
                self._format_percent(self.summary.get("weighted_precision")),
            ),
            ("Weighted Recall", self._format_percent(self.summary.get("weighted_recall"))),
            ("Weighted F1", self._format_percent(self.summary.get("weighted_f1"))),
            (
                "Macro Precision",
                self._format_percent(self.summary.get("macro_precision")),
            ),
            ("Macro Recall", self._format_percent(self.summary.get("macro_recall"))),
            ("Macro F1", self._format_percent(self.summary.get("macro_f1"))),
            ("Loss", self._format_number(self.summary.get("loss"), digits=4)),
            ("Test Images", self._format_int(self.summary.get("test_images"))),
            ("Classes", self._format_int(self.summary.get("num_classes"))),
        ]

        for index, (label, value) in enumerate(metric_cards):
            row = index // 4
            column = index % 4
            card_grid.addWidget(self._make_metric_card(label, value), row, column)

        layout.addWidget(cards)
        layout.addWidget(
            self._make_details_group(
                "Evaluation Details",
                [
                    ("Evaluated At", self.summary.get("evaluated_at", "--")),
                    (
                        "Python Executable",
                        self.summary.get("python_executable", "--"),
                    ),
                    ("Model Path", self.summary.get("model_path", "--")),
                    ("Dataset Root", self.summary.get("dataset_root", "--")),
                    (
                        "Original Image Root",
                        self.summary.get("original_image_root", "--"),
                    ),
                    ("Image Size", self.summary.get("image_size", "--")),
                    ("Batch Size", self.summary.get("batch_size", "--")),
                    ("Preprocessing", self.summary.get("preprocessing", "--")),
                    (
                        "Class Labels",
                        ", ".join(self.summary.get("class_labels", [])) or "--",
                    ),
                    (
                        "Class Folders",
                        ", ".join(self.summary.get("class_folders", [])) or "--",
                    ),
                ],
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Overall Metrics Chart", self.performance_dir / "metrics_overview.png"
            )
        )
        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _build_class_scores_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        layout.addWidget(
            self._make_details_group(
                "Class Summary Snapshot",
                [
                    (
                        "Best F1",
                        self._class_extreme_text("f1_score", highest=True),
                    ),
                    (
                        "Lowest F1",
                        self._class_extreme_text("f1_score", highest=False),
                    ),
                    (
                        "Best Recall",
                        self._class_extreme_text("recall", highest=True),
                    ),
                    (
                        "Lowest Recall",
                        self._class_extreme_text("recall", highest=False),
                    ),
                ],
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Class-wise Precision, Recall, and F1",
                self.performance_dir / "class_wise_metrics.png",
            )
        )
        layout.addWidget(
            self._make_table_group(
                "Class Summary Table",
                self.class_rows,
                preferred_order=[
                    "class_name",
                    "dataset_folder",
                    "train_count",
                    "val_count",
                    "test_count",
                    "predicted_count",
                    "precision",
                    "recall",
                    "f1_score",
                    "support",
                ],
            )
        )
        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _build_confusions_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        layout.addWidget(
            self._make_details_group(
                "Top Confusion Summary",
                [
                    (
                        "Most Frequent Error",
                        self._top_confusion_text(0),
                    ),
                    (
                        "Second Most Frequent Error",
                        self._top_confusion_text(1),
                    ),
                    (
                        "Third Most Frequent Error",
                        self._top_confusion_text(2),
                    ),
                ],
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Confusion Matrix (Counts)",
                self.performance_dir / "confusion_matrix_counts.png",
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Confusion Matrix (Row-Normalized)",
                self.performance_dir / "confusion_matrix_normalized.png",
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Top Misclassifications",
                self.performance_dir / "top_confusions.png",
            )
        )
        layout.addWidget(
            self._make_table_group(
                "Confusion Table",
                self.confusion_rows,
                preferred_order=[
                    "true_class",
                    "predicted_class",
                    "count",
                    "true_class_total",
                    "true_class_error_rate",
                ],
            )
        )
        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _build_distributions_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        layout.addWidget(
            self._make_image_group(
                "Dataset Distribution",
                self.performance_dir / "dataset_distribution.png",
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Prediction Distribution",
                self.performance_dir / "prediction_distribution.png",
            )
        )
        layout.addWidget(
            self._make_image_group(
                "Confidence Distribution",
                self.performance_dir / "confidence_distribution.png",
            )
        )
        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _build_samples_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        layout.addWidget(
            self._make_image_group(
                "Sample Predictions",
                self.performance_dir / "sample_predictions.png",
            )
        )
        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _build_reports_tab(self):
        body = QWidget()
        layout = QVBoxLayout(body)

        layout.addWidget(
            self._make_details_group(
                "Files Included",
                [
                    ("metrics_summary.json", self._exists_text("metrics_summary.json")),
                    ("class_summary.csv", self._exists_text("class_summary.csv")),
                    ("top_confusions.csv", self._exists_text("top_confusions.csv")),
                    ("report.txt", self._exists_text("report.txt")),
                    ("metrics_overview.png", self._exists_text("metrics_overview.png")),
                    (
                        "class_wise_metrics.png",
                        self._exists_text("class_wise_metrics.png"),
                    ),
                    (
                        "confusion_matrix_counts.png",
                        self._exists_text("confusion_matrix_counts.png"),
                    ),
                    (
                        "confusion_matrix_normalized.png",
                        self._exists_text("confusion_matrix_normalized.png"),
                    ),
                    (
                        "dataset_distribution.png",
                        self._exists_text("dataset_distribution.png"),
                    ),
                    (
                        "prediction_distribution.png",
                        self._exists_text("prediction_distribution.png"),
                    ),
                    (
                        "confidence_distribution.png",
                        self._exists_text("confidence_distribution.png"),
                    ),
                    ("sample_predictions.png", self._exists_text("sample_predictions.png")),
                    ("top_confusions.png", self._exists_text("top_confusions.png")),
                ],
            )
        )

        raw_json = QPlainTextEdit()
        raw_json.setReadOnly(True)
        raw_json.setMinimumHeight(220)
        raw_json.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #101010;"
            " border: 1px solid #2e2e2e;"
            " border-radius: 12px;"
            " padding: 10px;"
            " font-family: Consolas, 'Courier New', monospace;"
            "}"
        )
        raw_json.setPlainText(json.dumps(self.summary, indent=2) if self.summary else "{}")
        layout.addWidget(self._make_widget_group("metrics_summary.json", raw_json))

        report_box = QPlainTextEdit()
        report_box.setReadOnly(True)
        report_box.setMinimumHeight(420)
        report_box.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #101010;"
            " border: 1px solid #2e2e2e;"
            " border-radius: 12px;"
            " padding: 10px;"
            " font-family: Consolas, 'Courier New', monospace;"
            "}"
        )
        report_box.setPlainText(self.report_text or "report.txt not found.")
        layout.addWidget(self._make_widget_group("report.txt", report_box))

        layout.addStretch()

        return self._wrap_in_scroll(body)

    def _make_metric_card(self, title: str, value: str):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame {"
            " background-color: #171717;"
            " border: 1px solid #2f2f2f;"
            " border-radius: 14px;"
            "}"
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #9ea4ad; font-size: 12px;")
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f2f2f2;")

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        return card

    def _make_details_group(self, title: str, rows):
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        for key, value in rows:
            key_label = QLabel(f"{key}:")
            key_label.setStyleSheet("font-weight: bold; color: #d7d7d7;")
            value_label = QLabel(str(value))
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(key_label, value_label)

        return group

    def _make_image_group(self, title: str, image_path: Path):
        return self._make_widget_group(title, ScaledImageLabel(image_path))

    def _make_widget_group(self, title: str, widget: QWidget):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _make_table_group(self, title: str, rows, preferred_order=None):
        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setStyleSheet(
            "QTableWidget {"
            " background-color: #111111;"
            " alternate-background-color: #181818;"
            " gridline-color: #333333;"
            " border: 1px solid #2e2e2e;"
            " border-radius: 12px;"
            "}"
            "QHeaderView::section {"
            " background-color: #1f1f1f;"
            " color: white;"
            " border: none;"
            " padding: 6px;"
            " font-weight: bold;"
            "}"
        )

        if rows:
            columns = list(rows[0].keys())
            if preferred_order:
                ordered = [name for name in preferred_order if name in columns]
                ordered.extend([name for name in columns if name not in ordered])
                columns = ordered

            table.setColumnCount(len(columns))
            table.setRowCount(len(rows))
            table.setHorizontalHeaderLabels([self._humanize(name) for name in columns])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)

            for row_index, row in enumerate(rows):
                for col_index, column_name in enumerate(columns):
                    table.setItem(
                        row_index,
                        col_index,
                        self._make_table_item(self._format_cell_value(row.get(column_name))),
                    )
        else:
            table.setColumnCount(1)
            table.setRowCount(1)
            table.setHorizontalHeaderLabels(["Status"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setItem(0, 0, self._make_table_item("No data found"))

        return self._make_widget_group(title, table)

    def _wrap_in_scroll(self, body: QWidget):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        layout.addWidget(scroll)
        return container

    def _load_json(self, filename: str):
        file_path = self.performance_dir / filename
        if not file_path.exists():
            return {}

        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_csv(self, filename: str):
        file_path = self.performance_dir / filename
        if not file_path.exists():
            return []

        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError:
            return []

    def _load_text(self, filename: str):
        file_path = self.performance_dir / filename
        if not file_path.exists():
            return ""

        try:
            return file_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _exists_text(self, filename: str):
        return "Available" if (self.performance_dir / filename).exists() else "Missing"

    def _class_extreme_text(self, metric_name: str, highest: bool):
        if not self.class_rows:
            return "--"

        def metric_value(row):
            try:
                return float(row.get(metric_name, 0))
            except (TypeError, ValueError):
                return 0.0

        target = max(self.class_rows, key=metric_value) if highest else min(
            self.class_rows, key=metric_value
        )
        label = target.get("class_name", "--")
        return f"{label} ({self._format_percent(metric_value(target))})"

    def _top_confusion_text(self, index: int):
        if len(self.confusion_rows) <= index:
            return "--"

        row = self.confusion_rows[index]
        true_class = row.get("true_class", "--")
        predicted_class = row.get("predicted_class", "--")
        count = self._format_int(row.get("count"))
        error_rate = self._format_percent(row.get("true_class_error_rate"))
        return f"{true_class} -> {predicted_class} | {count} images | error rate {error_rate}"

    def _make_table_item(self, text: str):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _format_cell_value(self, value):
        if value is None or value == "":
            return "--"

        if isinstance(value, str):
            trimmed = value.strip()
            try:
                if trimmed.isdigit():
                    return f"{int(trimmed):,}"
                number = float(trimmed)
                if 0 <= number <= 1 and any(
                    key in trimmed.lower()
                    for key in ("0.", "1.0", "1.00", "1.000")
                ):
                    return f"{number:.4f}"
                if number.is_integer():
                    return f"{int(number):,}"
                return f"{number:.4f}"
            except ValueError:
                return trimmed

        return str(value)

    def _format_percent(self, value):
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "--"

    def _format_number(self, value, digits: int = 4):
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    def _format_int(self, value):
        try:
            return f"{int(float(value)):,}"
        except (TypeError, ValueError):
            return "--"

    def _humanize(self, name: str):
        return name.replace("_", " ").title()
