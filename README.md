# DrishtiAI 0.1

DrishtiAI is a desktop classroom analytics application built with `PySide6`, `OpenCV`, `TensorFlow`, and `face-recognition`.
It provides live camera-based monitoring with attendance tracking, multi-camera support, timetable integration, emotion analytics, and reporting workflows.

## Features

- Login-protected desktop app
- Live face tracking
- Multi-camera view
- Timetable-based auto attendance
- Attendance management and exports
- Emotion analytics dashboard
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
├─ gui/                        # Main GUI app and UI pages
│  ├─ main_gui.py              # App entry point
│  ├─ ui/                      # Dashboard, tracking, attendance, etc.
│  ├─ users.json               # Login users (hashed passwords)
│  └─ settings.json            # GUI settings (optional)
├─ features/                   # Tracking, timetable, and multi-camera logic
├─ attendance/                 # Attendance outputs
├─ reports/                    # Generated reports
├─ snapshots/                  # Captured snapshots
├─ known_faces/                # Face image database
├─ timetable/                  # Timetable files/data
├─ requirements-runtime.txt    # Runtime dependencies
├─ setup_drishtiai.bat         # One-time setup script
└─ run_drishtiai.bat           # Run script
```

## Prerequisites

1. Windows 10 or 11
2. Python `3.11` (recommended) or `3.10`
3. Python Launcher (`py`) available in terminal
4. Webcam/camera access enabled
5. Internet connection for first dependency install
6. (If needed) Microsoft C++ Build Tools for packages like `dlib` / `face-recognition`

## Setup (First Time)

Run this once from project root:

```bat
setup_drishtiai.bat
```

What this script does:

1. Verifies `gui/main_gui.py` exists
2. Selects Python `3.11` first, then `3.10`
3. Creates virtual env (`.venv311` or `.venv310`)
4. Upgrades `pip`, `setuptools`, `wheel`
5. Installs `requirements-runtime.txt`
6. Creates runtime folders if missing:
   - `attendance`
   - `reports`
   - `snapshots`
   - `timetable`
   - `known_faces`

## Face Env Clone Setup (Same as author machine)

If you want to reproduce the author's `face_env` package set on another Windows device, use:

- `setup_face_env_clone.bat`
- `requirements-face_env-lock.txt`

Run from project root:

```bat
setup_face_env_clone.bat
```

What this script does:

1. Detects Python (prefers `3.10`)
2. Creates `face_env` virtual environment
3. Activates `face_env`
4. Installs pinned packages from `requirements-face_env-lock.txt`
5. Saves installed package snapshot to `face_env\installed-freeze.txt`

Notes:

- Best compatibility target is Python `3.10.x` (original env is `3.10.11`).
- This clone setup is provided for environment parity with the original `face_env`.

## Run Process

After setup, start the app:

```bat
run_drishtiai.bat
```

Run script behavior:

1. Finds Python in `.venv311`, `.venv310`, or `.venv`
2. Sets `PYTHONPATH` to project root
3. Launches `gui/main_gui.py`

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
- `face_tolerance`
- `model_path`
- `attendance_path`
- `snapshot_path`
- `auto_attendance`
- `cameras` (multi-camera config)

## How to Use (Typical Flow)

1. Run setup once: `setup_drishtiai.bat`
2. Start app: `run_drishtiai.bat`
3. Create account (first time) or log in
4. Configure camera and model in Settings
5. Add/manage timetable in Time-Table Editor
6. Use Live Tracking / Multi Camera View for monitoring
7. Review attendance and analytics pages
8. Check generated files in `attendance/`, `reports/`, `snapshots/`

## Emotion Model Train (Emotion_model_train)

For training your own emotion model, use this repository:

- `https://github.com/Prem43-sm/Emotion_Model_train_for_DrishtiAI.git`

Note:

- Repository/page name is renamed from `msc-ai-project` to `Emotion_model_train`.

How to create and use your own model for DrishtiAI:

1. Clone/open the model training repo (`Emotion_model_train`).
2. Prepare your emotion dataset as required by that repo.
3. Run its training pipeline and export the trained model as `.h5`.
4. Rename/copy your trained file to `best_emotion_model.h5` (recommended) and place it in DrishtiAI project root.
5. Or keep your own filename and set it in DrishtiAI `settings.json`:
   - `model_path: "your_model_name.h5"`
6. Start DrishtiAI again using `run_drishtiai.bat`.

Required output for DrishtiAI runtime:

- Emotion model file in project root (`.h5`)
- Preferred name: `best_emotion_model.h5`

## Manual Run (Optional)

If needed, run manually without batch file:

```powershell
.\.venv311\Scripts\python.exe gui\main_gui.py
```

or

```powershell
.\.venv310\Scripts\python.exe gui\main_gui.py
```

## Troubleshooting

- Error: `No virtual environment found`
  - Run `setup_drishtiai.bat` first.

- Error: `Python 3.10/3.11 not found via py launcher`
  - Install Python 3.11 and ensure `py` works in terminal.

- Dependency install fails for `face-recognition` / `dlib`
  - Install Microsoft C++ Build Tools, then rerun setup.

- App opens but camera does not work
  - Check camera permissions in Windows settings.
  - Verify `camera_index` in `settings.json`.

- Model file issues
  - Ensure `.h5` model file exists in project root.
  - Update `model_path` in `settings.json` to correct filename.

## Notes

- This project is currently Windows-focused because setup/run are provided as `.bat` scripts.
- Keep virtual environments and large model files out of version control when sharing code.
