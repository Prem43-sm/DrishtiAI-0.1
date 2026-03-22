import cv2


def iter_camera_backends():
    seen = set()
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        backend = getattr(cv2, name, None)
        if backend is None or backend in seen:
            continue
        seen.add(backend)
        yield name, backend


def probe_camera(camera_id):
    for backend_name, backend in iter_camera_backends():
        cap = cv2.VideoCapture(camera_id, backend)
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
    for backend_name, backend in iter_camera_backends():
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            return cap, backend_name, backend

        cap.release()

    return None, None, None
