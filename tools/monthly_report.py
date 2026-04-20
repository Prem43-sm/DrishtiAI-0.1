import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR
if not (ROOT_DIR / "gui").exists() and (ROOT_DIR.parent / "gui").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.project_paths import ATTENDANCE_DIR, REPORTS_DIR, ensure_runtime_layout

ensure_runtime_layout()

month = input("Enter month (YYYY-MM): ")

files = sorted([f.name for f in ATTENDANCE_DIR.iterdir() if f.is_file() and f.name.startswith(month)])

if not files:
    print("No data found")
    exit()

all_data = []

for file in files:
    df = pd.read_csv(ATTENDANCE_DIR / file)
    df["Date"] = file.replace(".csv", "")
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# -----------------------------
# CREATE REGISTER FORMAT
# -----------------------------

students = final_df["Name"].unique()
dates = sorted(final_df["Date"].unique())

register = pd.DataFrame({"Name": students})
register.insert(0, "Id", range(1, len(register) + 1))
register["Course"] = "MSc-IT"

for date in dates:

    day_data = final_df[final_df["Date"] == date]

    status = []

    for student in register["Name"]:

        record = day_data[day_data["Name"] == student]

        if not record.empty:
            time = record.iloc[0]["Time"]
            status.append(f"P ({time})")
        else:
            status.append("A")

    register[date] = status

# -----------------------------
# SAVE EXCEL
# -----------------------------

report_file = REPORTS_DIR / f"{month}_report.xlsx"

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:
    register.to_excel(writer, sheet_name="Monthly Register", index=False)

print("Report generated:", report_file)
