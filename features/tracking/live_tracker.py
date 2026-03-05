from datetime import datetime

# name -> data
_locations = {}

LIVE_TIMEOUT = 5  # seconds


def update_location(name, camera, class_name):

    _locations[name] = {
        "camera": camera,
        "class": class_name,
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