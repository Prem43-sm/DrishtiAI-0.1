import os
import sys
from pathlib import Path


def project_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def resource_path(relative_path):
    return os.path.join(str(project_root()), relative_path)
