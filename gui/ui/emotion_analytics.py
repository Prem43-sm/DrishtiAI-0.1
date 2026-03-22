import calendar
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EmotionAnalyticsPage(QWidget):
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
        self.runtime_enabled = False
        self.refresh_timer = None

        self._build_ui()
        self._set_paused_state()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 10)
        main_layout.setSpacing(8)

        title = QLabel("Emotion Analytics")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#f2f2f2;")
        main_layout.addWidget(title)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Single Student", "All Students"])
        controls.addWidget(self.mode_selector)

        controls.addWidget(QLabel("Class:"))
        self.class_selector = QComboBox()
        self.class_selector.setMinimumWidth(200)
        controls.addWidget(self.class_selector)

        self.student_label = QLabel("Student:")
        controls.addWidget(self.student_label)

        self.student_selector = QComboBox()
        self.student_selector.setMinimumWidth(180)
        controls.addWidget(self.student_selector)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_data)
        controls.addWidget(self.refresh_btn)
        controls.addStretch()

        self.data_status = QLabel("")
        self.data_status.setStyleSheet("color:#cfcfcf;")
        controls.addWidget(self.data_status)
        main_layout.addLayout(controls)

        self.alert_label = QLabel("")
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("padding:6px; border-radius:4px;")
        main_layout.addWidget(self.alert_label)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(8)

        self.bar_fig = Figure(figsize=(4.5, 3.2), tight_layout=True)
        self.bar_canvas = FigureCanvas(self.bar_fig)
        charts_row.addWidget(self.bar_canvas, 1)

        self.pie_fig = Figure(figsize=(4.5, 3.2), tight_layout=True)
        self.pie_canvas = FigureCanvas(self.pie_fig)
        charts_row.addWidget(self.pie_canvas, 1)

        self.line_fig = Figure(figsize=(4.5, 3.2), tight_layout=True)
        self.line_canvas = FigureCanvas(self.line_fig)
        charts_row.addWidget(self.line_canvas, 1)

        main_layout.addLayout(charts_row, 1)

        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(34)
        self.status_bar.setStyleSheet(
            "QFrame { background:#161a1f; border:1px solid #2a2f36; border-radius:5px; }"
            "QLabel { color:#d7dde8; font-size:12px; }"
        )
        status_row = QHBoxLayout(self.status_bar)
        status_row.setContentsMargins(10, 4, 10, 4)
        status_row.setSpacing(14)
        self.today_time_label = QLabel("Start: --  End: --  Duration: --")
        self.today_emotion_label = QLabel("Today: --")
        self.today_emotion_label.setStyleSheet("color:#9fd2ff; font-size:12px;")
        status_row.addWidget(self.today_time_label)
        status_row.addWidget(self.today_emotion_label, 1)
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)

        self.mode_selector.currentIndexChanged.connect(self._on_filter_change)
        self.class_selector.currentIndexChanged.connect(self._on_filter_change)
        self.student_selector.currentIndexChanged.connect(self._on_filter_change)

    def _style_axes(self, ax):
        ax.set_facecolor("#0f141b")
        for spine in ax.spines.values():
            spine.set_color("#425167")
        ax.tick_params(colors="#d7dde8")
        ax.xaxis.label.set_color("#e6ecf5")
        ax.yaxis.label.set_color("#e6ecf5")
        ax.title.set_color("#f3f7ff")

    @staticmethod
    def _set_fig_bg(fig):
        fig.patch.set_facecolor("#0b0f14")

    def _draw_empty(self, fig, canvas, message):
        fig.clear()
        self._set_fig_bg(fig)
        ax = fig.add_subplot(111)
        self._style_axes(ax)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#c9d2e3")
        ax.set_axis_off()
        canvas.draw_idle()

    def load_data(self):
        if not self.runtime_enabled:
            self._set_paused_state()
            return

        self._last_signature = self._data_signature()
        if not self.attendance_dir.exists():
            self._set_empty_data("Attendance folder not found.")
            return

        self.csv_sources = self._discover_csv_sources()
        if not self.csv_sources:
            self._set_empty_data("No CSV files found in attendance/.")
            return

        frames = []
        for csv_file in self.csv_sources:
            part = self._read_emotion_rows(csv_file)
            if not part.empty:
                frames.append(part)

        if not frames:
            self._set_empty_data("No usable emotion rows found.")
            return

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            self._set_empty_data("No data available.")
            return

        df["student"] = df["student"].astype("category")
        df["emotion"] = df["emotion"].astype("category")
        df["class"] = df["class"].astype("category")

        self.raw_df = df
        self.analytics_df = df
        self._populate_class_options()
        self._apply_mode_visibility()
        self._on_filter_change()

        self.data_status.setText(
            f"Loaded {len(self.analytics_df)} rows from {len(self.csv_sources)} file(s)"
        )

    def _start_auto_refresh(self):
        if self.refresh_timer is None:
            self.refresh_timer = QTimer(self)
            self.refresh_timer.setInterval(5000)
            self.refresh_timer.timeout.connect(self._auto_refresh_tick)
        if not self.refresh_timer.isActive():
            self.refresh_timer.start()

    def _stop_auto_refresh(self):
        if self.refresh_timer is not None and self.refresh_timer.isActive():
            self.refresh_timer.stop()

    def set_runtime_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.runtime_enabled:
            return

        self.runtime_enabled = enabled
        self.refresh_btn.setEnabled(enabled)

        if enabled:
            self.load_data()
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()
            self._set_paused_state()

    def _auto_refresh_tick(self):
        if not self.runtime_enabled:
            return
        signature = self._data_signature()
        if signature != self._last_signature:
            self.load_data()

    def _data_signature(self):
        if not self.attendance_dir.exists():
            return None
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
        class_col = self._find_column(header_df.columns, ["class", "classname", "batch", "section"])
        date_col = self._find_column(header_df.columns, ["date", "day", "recorded_date"])
        time_col = self._find_column(header_df.columns, ["time", "recorded_time", "clock"])
        datetime_col = self._find_column(header_df.columns, ["datetime", "timestamp"])

        if not student_col or not emotion_col:
            return pd.DataFrame()

        usecols = [student_col, emotion_col]
        for c in [class_col, date_col, time_col, datetime_col]:
            if c and c not in usecols:
                usecols.append(c)

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

        if datetime_col:
            out["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        else:
            if date_col:
                date_series = df[date_col].astype(str).str.strip()
            else:
                fallback_date = self._infer_date_from_filename(csv_file.name)
                date_series = pd.Series(
                    [fallback_date.strftime("%Y-%m-%d") if pd.notna(fallback_date) else ""] * len(df)
                )
            if time_col:
                time_series = df[time_col].astype(str).str.strip()
                out["datetime"] = pd.to_datetime(date_series + " " + time_series, errors="coerce")
            else:
                out["datetime"] = pd.to_datetime(date_series, errors="coerce")

        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
        if out["date"].isna().all():
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

    def _populate_class_options(self):
        self.class_selector.blockSignals(True)
        current_class = self.class_selector.currentText().strip()
        self.class_selector.clear()
        self.class_selector.addItem("All Classes")

        data_classes = []
        if not self.analytics_df.empty:
            data_classes = self._clean_class_names(
                self.analytics_df["class"].astype(str).unique().tolist()
            )
        current_classes = self._get_current_classes()
        self.known_face_classes = current_classes
        classes = current_classes or data_classes
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

    def _view_filtered_df(self):
        class_df = self._class_filtered_df()
        if class_df.empty:
            return class_df
        if self.mode_selector.currentText().strip() == "Single Student":
            student = self.student_selector.currentText().strip()
            if student:
                return class_df[class_df["student"].astype(str) == student]
            return pd.DataFrame()
        return class_df

    def _on_filter_change(self):
        self._apply_mode_visibility()
        self._populate_student_options()
        self._update_all_views()

    def _apply_mode_visibility(self):
        single_mode = self.mode_selector.currentText().strip() == "Single Student"
        self.student_label.setVisible(single_mode)
        self.student_selector.setVisible(single_mode)

    def _update_all_views(self):
        view_df = self._view_filtered_df()
        self._update_bar_chart(view_df)
        self._update_pie_chart(view_df)
        self._update_line_chart(view_df)
        self._update_today_summary(view_df)
        self._update_alert_label(view_df)

    def _update_bar_chart(self, df):
        self.bar_fig.clear()
        self._set_fig_bg(self.bar_fig)
        ax = self.bar_fig.add_subplot(111)
        self._style_axes(ax)

        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", color="#c9d2e3")
            ax.set_axis_off()
            self.bar_canvas.draw_idle()
            return

        counts = df.groupby("emotion", observed=True).size().sort_values(ascending=False)
        ax.bar(counts.index.astype(str), counts.values, color="#4EA1FF", edgecolor="#89c2ff")
        ax.set_title("Bar")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=22)
        self.bar_canvas.draw_idle()

    def _update_pie_chart(self, df):
        self.pie_fig.clear()
        self._set_fig_bg(self.pie_fig)
        ax = self.pie_fig.add_subplot(111)
        self._style_axes(ax)

        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", color="#c9d2e3")
            ax.set_axis_off()
            self.pie_canvas.draw_idle()
            return

        counts = df.groupby("emotion", observed=True).size().sort_values(ascending=False)
        labels = counts.index.astype(str).tolist()
        values = counts.values

        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p >= 6 else "",
            pctdistance=0.72,
            startangle=140,
            wedgeprops={"linewidth": 0.8, "edgecolor": "#0b0f14"},
            textprops={"color": "#e8eef8", "fontsize": 9},
        )
        for t in autotexts:
            t.set_color("#f3f8ff")
            t.set_fontsize(8)

        # Legend avoids label collision on small chart area.
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            labelcolor="#d7dde8",
            fontsize=8,
        )
        ax.set_title("Pie")
        self.pie_canvas.draw_idle()

    def _update_line_chart(self, df):
        self.line_fig.clear()
        self._set_fig_bg(self.line_fig)
        ax = self.line_fig.add_subplot(111)
        self._style_axes(ax)

        if df.empty or "date" not in df.columns:
            ax.text(0.5, 0.5, "No current-month data", ha="center", va="center", color="#c9d2e3")
            ax.set_axis_off()
            self.line_canvas.draw_idle()
            return

        today = pd.Timestamp(datetime.now().date())
        month_mask = (
            df["date"].notna()
            & (df["date"].dt.year == today.year)
            & (df["date"].dt.month == today.month)
        )
        month_df = df.loc[month_mask, ["date", "emotion"]]
        if month_df.empty:
            ax.text(0.5, 0.5, "No current-month data", ha="center", va="center", color="#c9d2e3")
            ax.set_axis_off()
            self.line_canvas.draw_idle()
            return

        trend = (
            month_df.groupby(["date", "emotion"], observed=True)
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        if trend.empty:
            ax.text(0.5, 0.5, "No current-month data", ha="center", va="center", color="#c9d2e3")
            ax.set_axis_off()
            self.line_canvas.draw_idle()
            return

        days_in_month = calendar.monthrange(today.year, today.month)[1]
        day_index = pd.Index(range(1, days_in_month + 1), name="day")
        trend.index = trend.index.day
        trend = trend.groupby(level=0, sort=True).sum().reindex(day_index, fill_value=0)

        palette = ["#ff6b6b", "#ffd166", "#4ecdc4", "#5aa9ff", "#c792ea", "#95d5b2"]
        x_values = trend.index.to_list()
        for idx, emotion in enumerate(trend.columns):
            ax.plot(
                x_values,
                trend[emotion],
                label=str(emotion),
                linewidth=1.7,
                color=palette[idx % len(palette)],
            )

        ax.set_title("Line")
        ax.set_xlabel("Day")
        ax.set_ylabel("Events/Day")
        ax.grid(color="#273240", alpha=0.35, linewidth=0.8)
        tick_step = 1 if days_in_month <= 16 else 2 if days_in_month <= 24 else 3
        tick_positions = list(range(1, days_in_month + 1, tick_step))
        if tick_positions[-1] != days_in_month:
            tick_positions.append(days_in_month)
        ax.set_xlim(1, days_in_month)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(day) for day in tick_positions])
        ax.tick_params(axis="x", rotation=0)
        ax.legend(
            loc="upper left",
            fontsize=7,
            frameon=True,
            facecolor="#11161d",
            edgecolor="#2f3b4a",
            labelcolor="#d7dde8",
        )
        self.line_canvas.draw_idle()

    def _update_today_summary(self, view_df):
        today = pd.Timestamp(datetime.now().date())
        today_df = view_df.copy()
        if "date" in today_df.columns:
            today_df = today_df[today_df["date"] == today]
        else:
            today_df = pd.DataFrame()

        if today_df.empty:
            self.today_time_label.setText("Start: --  End: --  Duration: --")
            self.today_emotion_label.setText("Today: --")
            return

        counts = today_df.groupby("emotion", observed=True).size().sort_values(ascending=False)
        total = int(counts.sum())
        pct_parts = []
        for emotion, count in counts.items():
            pct = (float(count) / float(total) * 100.0) if total else 0.0
            pct_parts.append(f"{emotion} {pct:.1f}%")
        self.today_emotion_label.setText("Today: " + " | ".join(pct_parts))

        timed_today = today_df.dropna(subset=["datetime"]).copy()
        timed_today["datetime"] = pd.to_datetime(timed_today["datetime"], errors="coerce")
        timed_today = timed_today.dropna(subset=["datetime"])
        if timed_today.empty:
            self.today_time_label.setText("Start: --  End: --  Duration: --")
            return

        start_dt = timed_today["datetime"].min()
        end_dt = timed_today["datetime"].max()
        duration_text = self._format_duration(end_dt - start_dt)
        self.today_time_label.setText(
            f"Start: {start_dt.strftime('%H:%M:%S')}  "
            f"End: {end_dt.strftime('%H:%M:%S')}  "
            f"Duration: {duration_text}"
        )

    @staticmethod
    def _format_duration(delta):
        total_seconds = int(max(0, delta.total_seconds()))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if hours > 0:
            return f"{hours} h {minutes} min"
        return f"{minutes} min"

    def _update_alert_label(self, df):
        alert_text = self._find_sadness_alert(df)
        if alert_text:
            self.alert_label.setText(alert_text)
            self.alert_label.setStyleSheet(
                "padding:6px; border-radius:4px; background-color:#3a1212; color:#ff9c9c;"
            )
        else:
            self.alert_label.setText("No emotional risk detected")
            self.alert_label.setStyleSheet(
                "padding:6px; border-radius:4px; background-color:#10351c; color:#8bff9b;"
            )

    def _find_sadness_alert(self, df):
        if df.empty:
            return None
        sad_df = df[df["emotion"].astype(str).str.lower() == "sad"].copy()
        if sad_df.empty or "date" not in sad_df.columns:
            return None
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
                    return f"Alert: {student} showing sadness for 3 consecutive days"
        return None

    def _set_paused_state(self):
        message = "Motion analytics paused. Enable from Dashboard."
        self.raw_df = pd.DataFrame()
        self.analytics_df = pd.DataFrame()
        self._populate_class_options()
        self._apply_mode_visibility()
        self.data_status.setText(message)
        self.refresh_btn.setEnabled(False)
        self.today_time_label.setText("Start: --  End: --  Duration: --")
        self.today_emotion_label.setText("Today: --")
        self._update_alert_label(pd.DataFrame())

        self._draw_empty(self.bar_fig, self.bar_canvas, message)
        self._draw_empty(self.pie_fig, self.pie_canvas, "No data")
        self._draw_empty(self.line_fig, self.line_canvas, "No data")

    def _set_empty_data(self, message):
        self.raw_df = pd.DataFrame()
        self.analytics_df = pd.DataFrame()
        self._populate_class_options()
        self._apply_mode_visibility()
        self.data_status.setText(message)
        self.today_time_label.setText("Start: --  End: --  Duration: --")
        self.today_emotion_label.setText("Today: --")
        self._update_alert_label(pd.DataFrame())

        self._draw_empty(self.bar_fig, self.bar_canvas, message)
        self._draw_empty(self.pie_fig, self.pie_canvas, "No data")
        self._draw_empty(self.line_fig, self.line_canvas, "No data")

    def _get_known_face_classes(self):
        if not self.known_faces_dir.exists():
            return []
        return sorted([p.name for p in self.known_faces_dir.iterdir() if p.is_dir()])

    def _get_current_classes(self):
        classes = set(self._get_known_face_classes())
        if self.attendance_dir.exists():
            classes.update(p.name for p in self.attendance_dir.iterdir() if p.is_dir())
        return self._clean_class_names(classes)

    @staticmethod
    def _clean_class_names(values):
        cleaned = []
        seen = set()
        for value in values:
            name = str(value).strip()
            if not name or name.lower() in {"nan", "none", "null"}:
                continue
            if name not in seen:
                cleaned.append(name)
                seen.add(name)
        return sorted(cleaned)

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
