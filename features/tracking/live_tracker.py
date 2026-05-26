from datetime import datetime

# name -> data
_locations = {}

LIVE_TIMEOUT = 5  # seconds


def update_location(name, camera, class_name):
    name = str(name or "").strip()
    if not name or name == "Unknown":
        return

    _locations[name] = {
        "camera": str(camera or "Camera"),
        "class": str(class_name or "Unknown"),
        "time": datetime.now()
    }


def get_location(name):

    data = _locations.get(name)

    if not data:
        return None

    now = datetime.now()
    diff = (now - data["time"]).total_seconds()

    return {
        "name": name,
        "camera": data["camera"],
        "class": data["class"],
        "time": data["time"].strftime("%H:%M:%S"),
        "live": diff <= LIVE_TIMEOUT
    }

def get_all_locations():

    now = datetime.now()
    result = []

    for name, data in _locations.items():

        diff = (now - data["time"]).total_seconds()

        result.append({
            "name": name,
            "camera": data["camera"],
            "class": data["class"],
            "time": data["time"].strftime("%H:%M:%S"),
            "live": diff <= LIVE_TIMEOUT
        })

    return result


def get_camera_presence(class_name=None):
    now = datetime.now()
    selected_class = str(class_name or "").strip()
    cameras = {}

    for name, data in _locations.items():
        diff = (now - data["time"]).total_seconds()
        if diff > LIVE_TIMEOUT:
            continue
        if selected_class and selected_class != "Select Class":
            if str(data.get("class", "")) != selected_class:
                continue
        camera = str(data.get("camera", "Camera"))
        cameras.setdefault(camera, []).append(name)

    return {
        camera: sorted(names)
        for camera, names in sorted(cameras.items())
    }
