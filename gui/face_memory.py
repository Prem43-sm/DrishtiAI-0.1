import os

import face_recognition
import numpy as np


class FaceMemory:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceMemory()
        return cls._instance

    def __init__(self, folder="known_faces", tolerance=0.5):
        self.folder = folder
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

    def get_name(self, face_encoding):
        if not self.known_encodings:
            return "Unknown"

        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        if face_distances.size == 0:
            return "Unknown"

        best_match_index = int(np.argmin(face_distances))
        if float(face_distances[best_match_index]) <= self.tolerance:
            return self.known_names[best_match_index]

        return "Unknown"

    def reload(self):
        self.load_faces()
