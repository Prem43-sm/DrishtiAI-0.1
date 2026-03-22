import calendar
import json
import os
from datetime import datetime

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AttendancePage(QWidget):
    def __init__(self):
        super().__init__()

        self.base_dir = "attendance"
        self.face_dir = "known_faces"

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)

        self.current_view_df = pd.DataFrame()

        main_layout = QVBoxLayout()

        title = QLabel("Attendance")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        top_bar = QHBoxLayout()
        self.class_selector = QComboBox()
        self.load_classes()

        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDisplayFormat("dd-MM-yyyy")
        self.date_picker.setDate(QDate.currentDate())

        load_btn = QPushButton("Load Date")
        load_btn.clicked.connect(self.load_attendance)

        export_btn = QPushButton("Export Excel")
        export_btn.clicked.connect(self.export_excel)

        top_bar.addWidget(QLabel("Class:"))
        top_bar.addWidget(self.class_selector)
        top_bar.addWidget(QLabel("Date:"))
        top_bar.addWidget(self.date_picker)
        top_bar.addWidget(load_btn)
        top_bar.addStretch()
        top_bar.addWidget(export_btn)
        main_layout.addLayout(top_bar)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name...")
        self.search_input.textChanged.connect(self.filter_table)
        main_layout.addWidget(self.search_input)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)
        self.active_class = ""
        self.active_period = None

    # ---------------------------------------------------------
    # CLASS LIST
    # ---------------------------------------------------------
    def load_classes(self):
        current = self.class_selector.currentText()
        self.class_selector.clear()

        classes = []
        if os.path.exists(self.face_dir):
            for cls in sorted(os.listdir(self.face_dir)):
                full = os.path.join(self.face_dir, cls)
                if os.path.isdir(full):
                    classes.append(cls)
                    self.class_selector.addItem(cls)

        if current and current in classes:
            self.class_selector.setCurrentText(current)

    # ---------------------------------------------------------
    # SINGLE DATE VIEW (FULL CLASS)
    # ---------------------------------------------------------
    def load_attendance(self):
        cls = self.class_selector.currentText().strip()
        if not cls:
            QMessageBox.warning(self, "Error", "Select a class first")
            return

        date_str = self.date_picker.date().toString("yyyy-MM-dd")
        view_df = self._build_single_date_view(cls, date_str)
        self.current_view_df = view_df
        self._render_table(view_df)

    def _get_students_for_class(self, cls):
        students = set()
        class_dir = os.path.join(self.face_dir, cls)

        if os.path.exists(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".npy"):
                    students.add(os.path.splitext(file_name)[0])

        # fallback from attendance files if needed
        class_att_dir = os.path.join(self.base_dir, cls)
        if os.path.exists(class_att_dir):
            for file_name in os.listdir(class_att_dir):
                if not file_name.endswith(".csv"):
                    continue
                try:
                    tmp = pd.read_csv(os.path.join(class_att_dir, file_name))
                    if "Name" in tmp.columns:
                        for n in tmp["Name"].dropna().astype(str):
                            if n.strip():
                                students.add(n.strip())
                except Exception:
                    continue

        return sorted(students)

    def _get_periods_for_date(self, cls, date_str):
        periods = set()

        # 1) Prefer timetable day periods
        timetable_file = os.path.join("timetable", f"{cls}.json")
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = d.strftime("%A")
        except Exception:
            day_name = None

        if day_name and os.path.exists(timetable_file):
            try:
                with open(timetable_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for p in data.get("days", {}).get(day_name, []):
                    period = p.get("period")
                    if period is not None:
                        periods.add(str(period))
            except Exception:
                pass

        # 2) Merge file-based periods for that exact date
        class_path = os.path.join(self.base_dir, cls)
        if os.path.exists(class_path):
            for file_name in os.listdir(class_path):
                if not file_name.startswith(f"{date_str}_P") or not file_name.endswith(".csv"):
                    continue
                period = file_name.split("_P")[-1].replace(".csv", "").strip()
                if period:
                    periods.add(period)

        if not periods:
            periods = {"1"}

        return sorted(periods, key=lambda x: int(x) if str(x).isdigit() else 9999)

    def _build_single_date_view(self, cls, date_str):
        students = self._get_students_for_class(cls)
        periods = self._get_periods_for_date(cls, date_str)

        rows = []
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            selected_date = datetime.now().date()

        today = datetime.now().date()
        is_sunday = selected_date.weekday() == 6
        is_future = selected_date > today

        for name in students:
            row = {"Name": name, "Class": cls}
            for p in periods:
                col = f"P{p}"
                if is_future:
                    row[col] = ""
                elif is_sunday:
                    row[col] = "H"
                else:
                    row[col] = "A"
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Name", "Class"] + [f"P{p}" for p in periods])

        # fill present values
        class_path = os.path.join(self.base_dir, cls)
        for p in periods:
            file_path = os.path.join(class_path, f"{date_str}_P{p}.csv")
            if not os.path.exists(file_path):
                continue
            try:
                day_df = pd.read_csv(file_path)
            except Exception:
                continue

            if "Name" not in day_df.columns:
                continue

            for _, r in day_df.iterrows():
                n = str(r.get("Name", "")).strip()
                if not n:
                    continue
                if n not in set(df["Name"].astype(str)):
                    # include unknown/new name too
                    new_row = {"Name": n, "Class": cls}
                    for pp in periods:
                        col = f"P{pp}"
                        if is_future:
                            new_row[col] = ""
                        elif is_sunday:
                            new_row[col] = "H"
                        else:
                            new_row[col] = "A"
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                t = str(r.get("Time", "")).strip()
                mark = f"P ({t})" if t and t.lower() != "nan" else "P"
                df.loc[df["Name"].astype(str) == n, f"P{p}"] = mark

        if not df.empty:
            df = df.sort_values("Name").reset_index(drop=True)
        return df

    def _render_table(self, df):
        if df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))

    def filter_table(self):
        text = self.search_input.text().lower().strip()
        if self.current_view_df.empty:
            return
        if not text:
            self._render_table(self.current_view_df)
            return

        filtered = self.current_view_df[
            self.current_view_df["Name"].astype(str).str.lower().str.contains(text)
        ]
        self._render_table(filtered)

    def set_active_class(self, class_name, period=None):
        self.active_class = str(class_name or "").strip()
        self.active_period = period
        if not self.active_class:
            return

        if self.class_selector.findText(self.active_class) == -1:
            self.class_selector.addItem(self.active_class)
        self.class_selector.setCurrentText(self.active_class)

    def stop_auto_attendance(self):
        self.active_period = None

    # ---------------------------------------------------------
    # MONTHLY EXPORT (kept)
    # ---------------------------------------------------------
    def export_excel(self):
        cls = self.class_selector.currentText().strip()
        if not cls:
            QMessageBox.warning(self, "Error", "Select a class first")
            return

        class_path = os.path.join(self.base_dir, cls)
        if not os.path.exists(class_path):
            QMessageBox.warning(self, "Error", "No attendance folder for selected class")
            return

        monthly_groups = self._load_monthly_data(class_path, cls)
        if not monthly_groups:
            QMessageBox.warning(self, "Error", "No attendance data to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Monthly Report",
            f"{cls}_monthly_report.xlsx",
            "Excel (*.xlsx)",
        )
        if not path:
            return

        timetable_periods = self._get_timetable_periods(cls)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for month_key, month_df in sorted(monthly_groups.items()):
                students_df = self._get_all_students_for_class(cls, month_df)
                report_df = self._build_month_matrix(month_df, month_key, timetable_periods)
                report_df = self._ensure_all_students(report_df, students_df)
                if report_df.empty:
                    continue
                sheet_name = month_key.replace("-", "_")
                report_df.to_excel(writer, sheet_name=sheet_name, index=True)

        QMessageBox.information(self, "Done", "Monthly report exported successfully")

    def _load_monthly_data(self, class_path, cls):
        groups = {}

        for file_name in sorted(os.listdir(class_path)):
            if not file_name.endswith(".csv") or "_P" not in file_name:
                continue

            full = os.path.join(class_path, file_name)
            try:
                day_df = pd.read_csv(full)
            except Exception:
                continue

            if day_df.empty:
                continue

            date_part = file_name.split("_P")[0]
            try:
                date_obj = datetime.strptime(date_part, "%Y-%m-%d")
            except ValueError:
                continue

            period_str = file_name.split("_P")[-1].replace(".csv", "").strip()
            month_key = date_obj.strftime("%Y-%m")

            if "Name" not in day_df.columns:
                continue

            if "Class" not in day_df.columns:
                day_df["Class"] = cls

            if "Time" not in day_df.columns:
                day_df["Time"] = ""

            day_df["Date"] = date_obj.strftime("%Y-%m-%d")
            day_df["Period"] = str(period_str)
            day_df["Time"] = day_df["Time"].astype(str)

            groups.setdefault(month_key, []).append(day_df[["Name", "Class", "Time", "Date", "Period"]])

        for key in list(groups.keys()):
            groups[key] = pd.concat(groups[key], ignore_index=True)

        return groups

    def _get_all_students_for_class(self, class_name, month_df):
        names = set()
        class_face_dir = os.path.join(self.face_dir, class_name)
        if os.path.exists(class_face_dir):
            for file_name in os.listdir(class_face_dir):
                if file_name.endswith(".npy"):
                    names.add(os.path.splitext(file_name)[0])

        if not month_df.empty and "Name" in month_df.columns:
            for n in month_df["Name"].dropna().astype(str):
                if n.strip():
                    names.add(n.strip())

        if not names:
            return pd.DataFrame(columns=["Name", "Class"])

        return pd.DataFrame(
            [{"Name": n, "Class": class_name} for n in sorted(names)],
            columns=["Name", "Class"],
        )

    def _get_timetable_periods(self, class_name):
        timetable_file = os.path.join("timetable", f"{class_name}.json")
        periods = set()

        if os.path.exists(timetable_file):
            try:
                with open(timetable_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for _, day_periods in data.get("days", {}).items():
                    for item in day_periods:
                        p = item.get("period")
                        if p is not None:
                            periods.add(str(p))
            except Exception:
                pass

        if not periods:
            periods = {"1"}

        return sorted(periods, key=lambda x: int(x) if str(x).isdigit() else 9999)

    def _build_month_matrix(self, month_df, month_key, periods):
        year, month = [int(x) for x in month_key.split("-")]
        days_in_month = calendar.monthrange(year, month)[1]
        all_dates = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, days_in_month + 1)]

        students = month_df[["Name", "Class"]].dropna().drop_duplicates().sort_values(["Name", "Class"])
        if students.empty:
            return pd.DataFrame()

        idx = pd.MultiIndex.from_frame(students, names=["Name", "Class"])

        col_tuples = []
        for date_str in all_dates:
            for p in periods:
                col_tuples.append((date_str, f"P{p}"))

        cols = pd.MultiIndex.from_tuples(col_tuples, names=["Date", "Period"])
        report = pd.DataFrame("", index=idx, columns=cols)
        today = datetime.now().date()

        for date_str in all_dates:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            for p in periods:
                col_key = (date_str, f"P{p}")
                if d > today:
                    base_val = ""
                elif d.weekday() == 6:
                    base_val = "H"
                else:
                    base_val = "A"
                report[col_key] = base_val

        for _, row in month_df.iterrows():
            name = str(row.get("Name", "")).strip()
            cls = str(row.get("Class", "")).strip()
            date_str = str(row.get("Date", "")).strip()
            period = str(row.get("Period", "")).strip()
            time_str = str(row.get("Time", "")).strip()

            if not name or not cls or not date_str or not period:
                continue
            if (name, cls) not in report.index:
                continue

            col_key = (date_str, f"P{period}")
            if col_key not in report.columns:
                continue

            mark = f"P ({time_str})" if time_str and time_str.lower() != "nan" else "P"
            report.at[(name, cls), col_key] = mark

        return report

    def _ensure_all_students(self, report_df, students_df):
        if students_df.empty:
            return report_df

        if report_df.empty:
            idx = pd.MultiIndex.from_frame(students_df, names=["Name", "Class"])
            return pd.DataFrame(index=idx)

        existing = set(report_df.index.tolist())
        to_add = []
        for _, row in students_df.iterrows():
            key = (str(row["Name"]), str(row["Class"]))
            if key not in existing:
                to_add.append(key)

        if not to_add:
            return report_df

        add_df = pd.DataFrame(
            [["" for _ in range(len(report_df.columns))] for _ in range(len(to_add))],
            index=pd.MultiIndex.from_tuples(to_add, names=report_df.index.names),
            columns=report_df.columns,
        )

        today = datetime.now().date()
        for date_str, period in report_df.columns:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if d > today:
                base_val = ""
            elif d.weekday() == 6:
                base_val = "H"
            else:
                base_val = "A"
            add_df[(date_str, period)] = base_val

        out = pd.concat([report_df, add_df], axis=0)
        out = out.sort_index()
        return out

    def showEvent(self, event):
        self.load_classes()
        super().showEvent(event)
