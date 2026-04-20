import os

import cv2


def _backend_names(for_scan=False):
    # On Windows, DirectShow can emit noisy warnings or fail during index probing.
    # Prefer MSMF for discovery and keep DirectShow only as a last-resort open backend.
    if os.name == "nt":
        if for_scan:
            return ("CAP_MSMF", "CAP_ANY")
        return ("CAP_MSMF", "CAP_ANY", "CAP_DSHOW")
    return ("CAP_ANY",)


def iter_camera_backends(for_scan=False):
    seen = set()
    for name in _backend_names(for_scan=for_scan):
        backend = getattr(cv2, name, None)
        if backend is None or backend in seen:
            continue
        seen.add(backend)
        yield name, backend


def _create_capture(camera_id, backend_name, backend):
    try:
        if backend_name == "CAP_ANY":
            return cv2.VideoCapture(camera_id)
        return cv2.VideoCapture(camera_id, backend)
    except Exception:
        return None


def probe_camera(camera_id):
    for backend_name, backend in iter_camera_backends(for_scan=True):
        cap = _create_capture(camera_id, backend_name, backend)
        if cap is None:
            continue
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if ok:
                return backend_name, backend
        finally:
            cap.release()
    return None, None


def scan_camera_ids(max_scan_cameras, should_stop=None):
    connected = []
    for camera_id in range(max(0, int(max_scan_cameras))):
        if should_stop and should_stop():
            break
        backend_name, _ = probe_camera(camera_id)
        if backend_name:
            connected.append(camera_id)
    return connected


def open_camera_capture(camera_id):
    for backend_name, backend in iter_camera_backends(for_scan=False):
        cap = _create_capture(camera_id, backend_name, backend)
        if cap is None:
            continue
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            return cap, backend_name, backend

        cap.release()

    return None, None, None
