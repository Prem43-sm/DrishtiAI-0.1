import json
from gui.camera_worker import CameraWorker


class MultiCameraManager:

    def __init__(self):
        self.workers = []
        self.cameras = []
        self.load_config()

    # ---------------------------------------------------------

    def load_config(self):

        try:
            with open("settings.json", "r") as f:
                data = json.load(f)
                self.cameras = data.get("cameras", [])
        except:
            self.cameras = []

    # ---------------------------------------------------------

    def start_all(self):

        self.stop_all()  # 🔥 prevent duplicate workers

        print("Starting all cameras...")

        for cam in self.cameras:

            cam_id = cam.get("id", 0)
            cam_name = cam.get("name", "Camera")

            print(f"Starting Camera {cam_id} → {cam_name}")

            worker = CameraWorker(
                camera_id=cam_id,
                camera_name=cam_name
            )

            worker.start()
            self.workers.append(worker)

    # ---------------------------------------------------------

    def stop_all(self):

        if not self.workers:
            return

        print("Stopping all cameras...")

        for worker in self.workers:
            worker.stop()

        self.workers.clear()