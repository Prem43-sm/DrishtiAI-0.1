import pandas as pd
import os
from datetime import datetime

ATTENDANCE_DIR = "attendance"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ✅ AUTO CURRENT MONTH
month = datetime.now().strftime("%Y-%m")

files = sorted([f for f in os.listdir(ATTENDANCE_DIR) if f.startswith(month)])

if not files:
    print("No data found for", month)
    exit()

all_data = []

for file in files:
    path = os.path.join(ATTENDANCE_DIR, file)
    df = pd.read_csv(path)

    df["Date"] = file.replace(".csv", "")
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# =============================
# CREATE REGISTER FORMAT
# =============================

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

# =============================
# SAVE EXCEL
# =============================

report_file = os.path.join(REPORT_DIR, f"{month}_report.xlsx")

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:
    register.to_excel(writer, sheet_name="Monthly Register", index=False)

print("✅ Report generated:", report_file)