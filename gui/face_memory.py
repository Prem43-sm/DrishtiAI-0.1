import os

import face_recognition
import numpy as np

from core.project_paths import KNOWN_FACES_DIR, ensure_runtime_layout


class FaceMemory:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceMemory()
        return cls._instance

    def __init__(self, folder=None, tolerance=0.5):
        ensure_runtime_layout()
        self.folder = str(folder or KNOWN_FACES_DIR)
        self.tolerance = float(tolerance)
        self.known_encodings = []
        self.known_names = []
        os.makedirs(self.folder, exist_ok=True)
        self.load_faces()

    def load_faces(self):
        self.known_encodings.clear()
        self.known_names.clear()

        for root, _, files in os.walk(self.folder):
            for file in files:
                if not file.endswith(".npy"):
                    continue

                path = os.path.join(root, file)
                try:
                    encoding = np.load(path)
                except Exception:
                    continue

                self.known_encodings.append(encoding)
                self.known_names.append(os.path.splitext(file)[0])

        print(f"Faces Loaded: {len(self.known_names)}")

    def _similarity_threshold(self):
        return max(0.0, min(1.0, 1.0 - float(self.tolerance)))

    def _unknown_match(self):
        return {
            "name": "Unknown",
            "similarity": 0.0,
            "confidence": 0.0,
            "distance": None,
            "threshold": self._similarity_threshold(),
        }

    def match_face(self, face_encoding):
        if not self.known_encodings:
            return self._unknown_match()

        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        if face_distances.size == 0:
            return self._unknown_match()

        best_match_index = int(np.argmin(face_distances))
        distance = float(face_distances[best_match_index])
        similarity = max(0.0, 1.0 - distance)
        threshold = self._similarity_threshold()
        confidence_denominator = max(1e-6, 1.0 - threshold)
        confidence = max(
            0.0,
            min(1.0, (similarity - threshold) / confidence_denominator),
        )

        return {
            "name": self.known_names[best_match_index] if distance <= self.tolerance else "Unknown",
            "similarity": similarity,
            "confidence": confidence,
            "distance": distance,
            "threshold": threshold,
        }

    def get_name(self, face_encoding):
        return self.match_face(face_encoding)["name"]

    def reload(self):
        self.load_faces()
