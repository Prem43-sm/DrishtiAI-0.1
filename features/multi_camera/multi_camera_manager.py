from gui.camera_worker import CameraWorker
from gui.settings_manager import SettingsManager


class MultiCameraManager:
    def __init__(self):
        self.workers = []
        self.cameras = []
        self.load_config()

    def load_config(self):
        try:
            data = SettingsManager().load()
            self.cameras = data.get("cameras", [])
        except Exception:
            self.cameras = []

    def start_all(self):
        self.load_config()
        self.stop_all()

        print("Starting all cameras...")

        for cam in self.cameras:
            cam_id = cam.get("id", 0)
            cam_name = cam.get("name", "Camera")

            print(f"Starting Camera {cam_id} -> {cam_name}")

            worker = CameraWorker(camera_id=cam_id, camera_name=cam_name)
            worker.start()
            self.workers.append(worker)

    def stop_all(self):
        if not self.workers:
            return

        print("Stopping all cameras...")

        for worker in self.workers:
            worker.stop()

        self.workers.clear()
