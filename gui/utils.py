from core.project_paths import app_root as _app_root
from core.project_paths import resource_path as _resource_path
from core.project_paths import resource_root as _resource_root
from core.project_paths import resolve_app_path as _resolve_app_path


def project_root():
    return _resource_root()


def resource_path(relative_path):
    return str(_resource_path(relative_path))


def writable_root():
    return _app_root()


def writable_path(relative_path):
    return str(_resolve_app_path(relative_path))
