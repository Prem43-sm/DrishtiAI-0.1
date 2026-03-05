from pathlib import Path
import re

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


class EmotionAnalyticsPage(QWidget):
    # ---------------------------------------------------------
    # PAGE INIT
    # ---------------------------------------------------------
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parents[2]
        self.attendance_dir = self.base_dir / "attendance"
        self.known_faces_dir = self.base_dir / "known_faces"

        self.raw_df = pd.DataFrame()
        self.analytics_df = pd.DataFrame()
        self.csv_sources = []
        self.known_face_classes = []
        self._last_signature = None

        self._build_ui()
        self.load_data()
        self._start_auto_refresh()

    # ---------------------------------------------------------
    # UI BUILD
    # ---------------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        title = QLabel("Emotion Analytics")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        # Top filter bar
        controls = QHBoxLayout()
        controls.setSpacing(10)

        controls.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Single Student", "All Students"])
        controls.addWidget(self.mode_selector)

        controls.addWidget(QLabel("Class:"))
        self.class_selector = QComboBox()
        self.class_selector.setMinimumWidth(220)
        controls.addWidget(self.class_selector)

        self.student_label = QLabel("Student:")
        controls.addWidget(self.student_label)

        self.student_selector = QComboBox()
        self.student_selector.setMinimumWidth(220)
        controls.addWidget(self.student_selector)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_data)
        controls.addWidget(self.refresh_btn)
        controls.addStretch()

        self.data_status = QLabel("")
        controls.addWidget(self.data_status)

        main_layout.addLayout(controls)

        self.alert_label = QLabel("")
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("padding: 8px; border-radius: 4px;")
        main_layout.addWidget(self.alert_label)

        chart_row = QHBoxLayout()
        chart_row.setSpacing(12)

        self.student_fig = Figure(figsize=(6, 4), tight_layout=True)
        self.student_canvas = FigureCanvas(self.student_fig)
        chart_row.addWidget(self.student_canvas, 1)

        self.heatmap_fig = Figure(figsize=(6, 4), tight_layout=True)
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        chart_row.addWidget(self.heatmap_canvas, 1)

        main_layout.addLayout(chart_row, 1)
        self.setLayout(main_layout)

        self.mode_selector.currentIndexChanged.connect(self._on_filter_change)
        self.class_selector.currentIndexChanged.connect(self._on_filter_change)
        self.student_selector.currentIndexChanged.connect(self._on_filter_change)

    # ---------------------------------------------------------
    # DATA LOAD + PREP
    # ---------------------------------------------------------
    def load_data(self):
        self._last_signature = self._data_signature()

        if not self.attendance_dir.exists():
            self._set_empty_data("Attendance folder not found")
            return

        self.csv_sources = self._discover_csv_sources()
        if not self.csv_sources:
            self._set_empty_data("No CSV files found in attendance/")
            return

        frames = []
        for csv_file in self.csv_sources:
            df_part = self._read_emotion_rows(csv_file)
            if not df_part.empty:
                frames.append(df_part)

        if not frames:
            self._set_empty_data(
                "No emotion rows yet. Start live tracking to generate attendance/emotion_data.csv"
            )
            return

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            self._set_empty_data("No data available")
            return

        # Categories reduce memory for larger files.
        df["student"] = df["student"].astype("category")
        df["emotion"] = df["emotion"].astype("category")
        df["class"] = df["class"].astype("category")

        self.raw_df = df.copy()
        self.analytics_df = df
        self._populate_class_options()
        self._apply_mode_visibility()
        self._on_filter_change()

        self.data_status.setText(
            f"Loaded {len(self.analytics_df)} rows from {len(self.csv_sources)} file(s)"
        )

    def _start_auto_refresh(self):
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.timeout.connect(self._auto_refresh_tick)
        self.refresh_timer.start()

    def _auto_refresh_tick(self):
        signature = self._data_signature()
        if signature != self._last_signature:
            self.load_data()

    def _data_signature(self):
        if not self.attendance_dir.exists():
            return None

        emotion_file = self.attendance_dir / "emotion_data.csv"
        if emotion_file.exists():
            st = emotion_file.stat()
            return ("emotion_data.csv", st.st_size, st.st_mtime_ns)

        parts = []
        for csv_file in sorted(self.attendance_dir.rglob("*.csv")):
            try:
                st = csv_file.stat()
                parts.append((str(csv_file), st.st_size, st.st_mtime_ns))
            except OSError:
                continue
        return tuple(parts)

    def _discover_csv_sources(self):
        preferred = self.attendance_dir / "emotion_data.csv"
        if preferred.exists():
            return [preferred]
        return sorted(self.attendance_dir.rglob("*.csv"))

    def _read_emotion_rows(self, csv_file):
        try:
            header_df = pd.read_csv(csv_file, nrows=0)
        except Exception:
            return pd.DataFrame()

        student_col = self._find_column(
            header_df.columns,
            ["name", "student", "student_name", "studentname", "user", "userid"],
        )
        emotion_col = self._find_column(header_df.columns, ["emotion", "mood", "expression"])
        date_col = self._find_column(
            header_df.columns,
            ["date", "day", "recorded_date", "timestamp", "datetime"],
        )
        class_col = self._find_column(
            header_df.columns,
            ["class", "classname", "batch", "section"],
        )

        if not student_col or not emotion_col:
            return pd.DataFrame()

        usecols = [student_col, emotion_col]
        if date_col:
            usecols.append(date_col)
        if class_col:
            usecols.append(class_col)

        try:
            df = pd.read_csv(csv_file, usecols=usecols)
        except Exception:
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "student": df[student_col].astype(str).str.strip(),
                "emotion": df[emotion_col].astype(str).str.strip(),
            }
        )
        out = out[(out["student"] != "") & (out["emotion"] != "")]
        if out.empty:
            return pd.DataFrame()

        if class_col:
            out["class"] = df[class_col].astype(str).str.strip()
            out.loc[out["class"] == "", "class"] = self._infer_class_name(csv_file)
        else:
            out["class"] = self._infer_class_name(csv_file)

        if date_col:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        else:
            out["date"] = self._infer_date_from_filename(csv_file.name)

        return out.reset_index(drop=True)

    @staticmethod
    def _find_column(columns, candidates):
        normalized = {
            str(c).strip().lower().replace(" ", "").replace("_", ""): c
            for c in columns
        }
        for name in candidates:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in normalized:
                return normalized[key]
        return None

    def _infer_class_name(self, csv_file):
        if csv_file.parent == self.attendance_dir:
            return "General"
        return csv_file.parent.name

    @staticmethod
    def _infer_date_from_filename(filename):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if not match:
            return pd.NaT
        return pd.to_datetime(match.group(1), errors="coerce")

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------
    def _populate_class_options(self):
        self.class_selector.blockSignals(True)
        current_class = self.class_selector.currentText().strip()
        self.class_selector.clear()
        self.class_selector.addItem("All Classes")

        data_classes = []
        if not self.analytics_df.empty:
            data_classes = self.analytics_df["class"].astype(str).unique().tolist()

        self.known_face_classes = self._get_known_face_classes()
        classes = sorted(set(data_classes) | set(self.known_face_classes))
        self.class_selector.addItems(classes)

        if current_class and current_class in classes:
            self.class_selector.setCurrentText(current_class)
        self.class_selector.blockSignals(False)
        self._populate_student_options()

    def _populate_student_options(self):
        self.student_selector.blockSignals(True)
        current_student = self.student_selector.currentText().strip()
        self.student_selector.clear()

        filtered = self._class_filtered_df()
        if not filtered.empty:
            students = sorted(filtered["student"].astype(str).unique().tolist())
        else:
            selected_class = self.class_selector.currentText().strip()
            students = self._get_known_face_students(selected_class)

        self.student_selector.addItems(students)
        if current_student and current_student in students:
            self.student_selector.setCurrentText(current_student)

        self.student_selector.blockSignals(False)

    def _class_filtered_df(self):
        if self.analytics_df.empty:
            return pd.DataFrame()
        selected_class = self.class_selector.currentText().strip()
        if not selected_class or selected_class == "All Classes":
            return self.analytics_df
        return self.analytics_df[self.analytics_df["class"].astype(str) == selected_class]

    def _on_filter_change(self):
        self._apply_mode_visibility()
        self._populate_student_options()
        self._update_all_views()

    def _apply_mode_visibility(self):
        mode = self.mode_selector.currentText().strip()
        single_mode = mode == "Single Student"
        self.student_label.setVisible(single_mode)
        self.student_selector.setVisible(single_mode)

    # ---------------------------------------------------------
    # CHART + ALERT UPDATE
    # ---------------------------------------------------------
    def _update_all_views(self):
        class_df = self._class_filtered_df()
        mode = self.mode_selector.currentText().strip()

        if mode == "Single Student":
            student = self.student_selector.currentText().strip()
            if student:
                student_df = class_df[class_df["student"].astype(str) == student]
            else:
                student_df = pd.DataFrame()
            self._update_student_chart(student_df, student_name=student)
            self._update_alert_label(student_df, single_student=student)
        else:
            self._update_student_chart(class_df, student_name=None)
            self._update_alert_label(class_df, single_student=None)

        self._update_heatmap(class_df)

    def _update_student_chart(self, df, student_name=None):
        self.student_fig.clear()
        ax = self.student_fig.add_subplot(111)

        if df.empty:
            text = "No data available"
            if student_name:
                text = "Student not found"
            ax.text(0.5, 0.5, text, ha="center", va="center")
            ax.set_axis_off()
            self.student_canvas.draw_idle()
            return

        counts = (
            df.groupby("emotion", observed=True)
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )

        x_vals = counts["emotion"].astype(str).tolist()
        y_vals = counts["count"].tolist()
        ax.bar(x_vals, y_vals, color="#3E7CB1")
        if student_name:
            ax.set_title(f"Emotion Frequency - {student_name}")
        else:
            ax.set_title("Emotion Frequency - Selected Class")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        self.student_canvas.draw_idle()

    def _update_heatmap(self, class_df):
        self.heatmap_fig.clear()
        ax = self.heatmap_fig.add_subplot(111)

        if class_df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()
            self.heatmap_canvas.draw_idle()
            return

        heatmap_df = class_df.pivot_table(
            index="student",
            columns="emotion",
            aggfunc="size",
            fill_value=0,
        )
        if heatmap_df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()
            self.heatmap_canvas.draw_idle()
            return

        if sns is None:
            ax.text(
                0.5,
                0.5,
                "Seaborn is not installed.\nInstall seaborn to view heatmap.",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self.heatmap_canvas.draw_idle()
            return

        sns.heatmap(
            heatmap_df,
            ax=ax,
            cmap="YlOrRd",
            cbar=True,
            linewidths=0.3,
            linecolor="white",
        )
        ax.set_title("Class-wise Emotion Heatmap")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Student Names")
        self.heatmap_canvas.draw_idle()

    # ---------------------------------------------------------
    # ALERT SYSTEM
    # ---------------------------------------------------------
    def _update_alert_label(self, df, single_student=None):
        alert_text = self._find_sadness_alert(df, single_student)
        if alert_text:
            self.alert_label.setText(alert_text)
            self.alert_label.setStyleSheet(
                "padding: 8px; border-radius: 4px; background-color: #3a1010; color: #ff8a8a;"
            )
        else:
            self.alert_label.setText("No emotional risk detected")
            self.alert_label.setStyleSheet(
                "padding: 8px; border-radius: 4px; background-color: #103a1b; color: #8bff9b;"
            )

    def _find_sadness_alert(self, df, single_student=None):
        if df.empty:
            return None

        sad_df = df[df["emotion"].astype(str).str.lower() == "sad"].copy()
        if sad_df.empty:
            return None

        sad_df = sad_df.dropna(subset=["date"])
        if sad_df.empty:
            return None

        sad_df["date"] = pd.to_datetime(sad_df["date"], errors="coerce").dt.normalize()
        sad_df = sad_df.dropna(subset=["date"])
        if sad_df.empty:
            return None

        unique_days = (
            sad_df[["student", "date"]]
            .drop_duplicates()
            .sort_values(["student", "date"])
            .reset_index(drop=True)
        )

        for student, group in unique_days.groupby("student", observed=True):
            dates = group["date"].tolist()
            streak = 1
            for idx in range(1, len(dates)):
                day_gap = (dates[idx] - dates[idx - 1]).days
                streak = streak + 1 if day_gap == 1 else 1
                if streak >= 3:
                    student_text = str(student)
                    return f"\u26a0 Alert: {student_text} showing sadness for 3 consecutive days"

        if single_student:
            return None
        return None

    # ---------------------------------------------------------
    # EMPTY/ERROR STATE
    # ---------------------------------------------------------
    def _set_empty_data(self, message):
        self.raw_df = pd.DataFrame()
        self.analytics_df = pd.DataFrame()
        self._populate_class_options()
        self._apply_mode_visibility()
        self.data_status.setText(message)
        self._update_alert_label(pd.DataFrame(), single_student=None)

        self.student_fig.clear()
        ax1 = self.student_fig.add_subplot(111)
        ax1.text(0.5, 0.5, message, ha="center", va="center")
        ax1.set_axis_off()
        self.student_canvas.draw_idle()

        self.heatmap_fig.clear()
        ax2 = self.heatmap_fig.add_subplot(111)
        ax2.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax2.set_axis_off()
        self.heatmap_canvas.draw_idle()

    # ---------------------------------------------------------
    # KNOWN FACES FALLBACK (MATCH DATABASE PAGE)
    # ---------------------------------------------------------
    def _get_known_face_classes(self):
        if not self.known_faces_dir.exists():
            return []
        classes = []
        for item in sorted(self.known_faces_dir.iterdir()):
            if item.is_dir():
                classes.append(item.name)
        return classes

    def _get_known_face_students(self, selected_class):
        if not self.known_faces_dir.exists():
            return []

        students = set()
        if selected_class and selected_class != "All Classes":
            class_dirs = [self.known_faces_dir / selected_class]
        else:
            class_dirs = [p for p in self.known_faces_dir.iterdir() if p.is_dir()]

        for class_dir in class_dirs:
            if not class_dir.exists() or not class_dir.is_dir():
                continue
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".npy":
                    students.add(file_path.stem)

        return sorted(students)
