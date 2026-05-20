# DrishtiAI 0.1
<img width="971" height="485" alt="logo08888" src="https://github.com/user-attachments/assets/66e3eade-2cd9-4268-9a3d-4770cdd9376f" />


DrishtiAI is a desktop classroom analytics application built with `PySide6`, `OpenCV`, `TensorFlow`, and `face-recognition`.
It provides live camera-based monitoring with attendance tracking, multi-camera support, timetable integration, emotion analytics, and reporting workflows.

## Features

- Login-protected desktop app
- Live face tracking
- Multi-camera view
- Camera backend probing for more reliable webcam detection on Windows
- Timetable-based auto attendance
- Attendance management and exports
- Emotion analytics dashboard
- Runtime emotion model reload support
- Noise and misbehavior monitoring page
- Database/records page
- Model score/performance page
- Training page and app settings

## Tech Stack

- Python `3.10` or `3.11`
- PySide6 (`<6.10`) for GUI
- OpenCV for camera/video
- TensorFlow `2.16.1` for model loading/inference
- face-recognition for face detection/identity matching
- NumPy, Pandas, Matplotlib, scikit-learn

## Project Structure

```text
DrishtiAI 0.1/
├─ core/                       # Shared path/runtime helpers
├─ models/                     # Emotion + face-recognition model assets
├─ storage/                    # Runtime data (attendance, faces, reports, snapshots)
├─ datasets/                   # Local training/evaluation datasets and pipeline assets
├─ tools/                      # Utility scripts for evaluation and reports
├─ archive/                    # Legacy models and old experiments
├─ gui/                        # Main GUI app and UI pages
│  ├─ main_gui.py              # App entry point
│  ├─ camera_backend.py        # Camera backend probing/helpers
│  ├─ ui/                      # Dashboard, tracking, attendance, etc.
│  ├─ assets/branding/         # App logos/icons
│  └─ users.json               # Login users (hashed passwords)
├─ features/                   # Tracking, timetable, and multi-camera logic
├─ settings.json               # Main runtime config used by the app
├─ requirements-face_env-lock.txt # Locked setup dependencies
├─ requirements-runtime.txt    # Runtime dependencies
├─ setup_drishtiai.bat         # One-time setup script
└─ run_drishtiai.bat           # Run script
```

## Prerequisites

1. Windows 10 or 11
2. Python `3.10` for `setup_drishtiai.bat`
3. Python Launcher (`py`) available in terminal
4. Webcam/camera access enabled
5. Internet connection for first dependency install
6. (If needed) Microsoft C++ Build Tools for packages like `dlib` / `face-recognition`

Notes:

- `run_drishtiai.bat` can still start the app from an existing `.venv311`, `.venv310`, or `.venv` environment.
- For the bundled setup flow, the project expects Python `3.10.x`.

## Setup (First Time)

Run this once from project root:

```bat
setup_drishtiai.bat
```

What this script does:

1. Verifies `gui/main_gui.py` and `requirements-face_env-lock.txt` exist
2. Requires Python `3.10`
3. Creates virtual env `face_env`
4. Upgrades `pip`, `setuptools`, `wheel`
5. Installs locked packages from `requirements-face_env-lock.txt`
6. Creates required project folders if missing:
   - `storage/attendance`
   - `storage/reports`
   - `storage/snapshots`
   - `storage/snapshots/noise_alerts`
   - `storage/timetable`
   - `storage/known_faces`
   - `models/emotion`
   - `models/emotion/legacy`
   - `models/face_recognition`
   - `tools`, `archive`, `datasets`
7. Saves a package snapshot to `face_env\installed-freeze.txt`

Notes:

- Best compatibility target is Python `3.10.x` (original env is `3.10.11`).
- `setup_face_env_clone.bat` is still included as an alternate clone helper, but `setup_drishtiai.bat` now sets up the same `face_env` directly.

## Run Process

After setup, start the app:

```bat
run_drishtiai.bat
```

Run script behavior:

1. Finds Python in `face_env` first
2. Falls back to `.venv311`, `.venv310`, or `.venv` if needed
3. Sets `PYTHONPATH` to project root
4. Launches `gui/main_gui.py`

## First Login / Account Setup

On app start, login dialog opens with:

1. `Create Account`
2. `Log In`

To create an account:

1. Enter Secret Key: `CNA#123`
2. Enter User ID (3-32 chars)
3. Enter Password (8-12 chars)

Credentials are stored as SHA-256 hashes in `gui/users.json`.

## Configuration

Main runtime config file:

- `settings.json` (project root)

Important keys include:

- `camera_index`
- `resolution`
- `fps`
- `process_frame`
- `recognition_frames`
- `emotion_frames`
- `face_tolerance`
- `model_path`
- `attendance_path`
- `snapshot_path`
- `auto_attendance`
- `cameras` (multi-camera config)

Committed `settings.json` currently includes:

- `resolution`: `1920x1080`
- `fps`: `30`
- `process_frame`: `2`
- `recognition_frames`: `1`
- `emotion_frames`: `1`
- `model_path`: `models/emotion/step11_high_accuracy/final_model.h5`

If `settings.json` is deleted or reset, `gui/settings_manager.py` recreates factory defaults with:

- `resolution`: `640x480`
- `fps`: `30`
- `emotion_frames`: `10`

## How to Use (Typical Flow)

1. Run setup once: `setup_drishtiai.bat`
2. Start app: `run_drishtiai.bat`
3. Create account (first time) or log in
4. Configure camera and model in Settings
5. Add/manage timetable in Time-Table Editor
6. Use Live Tracking / Multi Camera View for monitoring
7. Review attendance and analytics pages
8. Check generated files under `storage/`
   - Examples: `storage/attendance`, `storage/reports`, `storage/snapshots`

## Emotion Model Train (Emotion_model_train)

For training your own emotion model, use this repository:

- `https://github.com/Prem43-sm/Emotion_Model_train_for_DrishtiAI.git`

Note:

- Repository/page name is renamed from `msc-ai-project` to `Emotion_model_train`.

How to create and use your own model for DrishtiAI:

1. Clone/open the model training repo (`Emotion_model_train`).
2. Prepare your emotion dataset as required by that repo.
3. Run its training pipeline and export the trained model as `.h5`.
4. Copy your trained model folder or model file into the project.
5. Preferred project location for this app is:
   - `models/emotion/step11_high_accuracy/final_model.h5`
   - with sidecar metadata files like `class_names.json`, `class_indices.json`, and `training_config.json`
6. Or keep your own filename and set it in DrishtiAI `settings.json`:
   - `model_path: "models/emotion/your_model/final_model.h5"`
7. Start DrishtiAI again using `run_drishtiai.bat`.

Required output for DrishtiAI runtime:

- Emotion model file path configured in `settings.json`
- Best compatibility comes from placing metadata next to the model file
- Current default runtime path:
  - `models/emotion/step11_high_accuracy/final_model.h5`
  - metadata in the same folder describes classes: `Angry`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`
  - input size: `160x160`
  - backbone: `EfficientNetV2B1`
  - reported test accuracy from imported metrics: `0.8613`
- The `.h5` model file itself is expected locally and is ignored by Git by default.

Local training helpers, datasets, and archived experiments are organized under `tools/`, `datasets/`, and `archive/`.
If you share the code, keep large local datasets and model artifacts out of version control.

## Manual Run (Optional)

If needed, run manually without batch file:

```powershell
.\face_env\Scripts\python.exe gui\main_gui.py
```

or

```powershell
.\.venv311\Scripts\python.exe gui\main_gui.py
```

## Troubleshooting

- Error: `No virtual environment found`
  - Run `setup_drishtiai.bat` first.

- Error: `Python 3.10 was not found`
  - Install Python 3.10.x and ensure `py -3.10` or `python` works in terminal.

- Dependency install fails for `face-recognition` / `dlib`
  - Install Microsoft C++ Build Tools, then rerun setup.

- App opens but camera does not work
  - Check camera permissions in Windows settings.
  - Verify `camera_index` in `settings.json`.

- Model file issues
  - Ensure the configured `.h5` model file exists at the path set in `settings.json`.
  - Update `model_path` in `settings.json` to correct filename.

## Notes

- This project is currently Windows-focused because setup/run are provided as `.bat` scripts.
- Keep virtual environments, large model files, image datasets, and local training/pipeline folders out of version control when sharing code.

## AI Analytics Additions

New integrated pages:

- `Emotion Performance Analytics`: monthly student emotion percentages, emotion score, engagement score, stability score, final performance score, charts, AI recommendations, and PDF/CSV/Excel exports.
- `Focus Mode Monitoring`: live webcam focus tracking with MediaPipe Face Mesh when installed, focus scoring, movement/sleeping/looking-away counters, database saving, and PDF/CSV/Excel exports.
- `Database -> AI Analytics Toolbox`: report tables, charts, monthly analytics, student comparison, export-folder access, and AI insights.

Analytics data is stored in:

- `storage/drishtiai_analytics.db`
- `emotion_reports`
- `focus_reports`

Generated analytics exports are saved under:

- `storage/reports/analytics`

Additional package notes:

- `mediapipe` enables landmark-based focus tracking.
- `openpyxl` is required for Excel exports.
- PDF export uses the existing Matplotlib stack.

## Smart Sidebar Workspace

The main window now includes a responsive smart sidebar system:

- Expanded, collapsed, hidden, overlay, and hover-expand navigation modes.
- A left menu rail appears when the sidebar is hidden so page titles and content do not overlap the menu button.
- `Focus Workspace` mode is available on Live Tracking, Emotion Analytics, and Focus Mode Monitoring pages.
- Sidebar preferences are saved locally in `settings.json` and can be changed from Settings:
  - Auto Hide Sidebar
  - Hover Expand
  - Overlay Navigation
  - Compact Icons
  - Fullscreen Workspace
