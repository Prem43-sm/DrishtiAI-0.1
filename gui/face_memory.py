import os
import numpy as np
import face_recognition


class FaceMemory:

    _instance = None   # 🔥 singleton

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceMemory()
        return cls._instance

    # ---------------------------------------------------------

    def __init__(self, folder="known_faces"):

        self.folder = folder

        self.known_encodings = []
        self.known_names = []

        os.makedirs(self.folder, exist_ok=True)

        self.load_faces()

    # ---------------------------------------------------------
    # 🔄 LOAD ALL FACES (CLASS-WISE SUPPORT)
    # ---------------------------------------------------------

    def load_faces(self):

        self.known_encodings.clear()
        self.known_names.clear()

        for root, dirs, files in os.walk(self.folder):

            for file in files:

                if not file.endswith(".npy"):
                    continue

                path = os.path.join(root, file)

                try:
                    encoding = np.load(path)

                    name = os.path.splitext(file)[0]

                    self.known_encodings.append(encoding)
                    self.known_names.append(name)

                except:
                    pass

        print(f"✅ Faces Loaded: {len(self.known_names)}")

    # ---------------------------------------------------------
    # 🔍 FIND NAME
    # ---------------------------------------------------------

    def get_name(self, face_encoding):

        if len(self.known_encodings) == 0:
            return "Unknown"

        matches = face_recognition.compare_faces(
            self.known_encodings,
            face_encoding,
            tolerance=0.5
        )

        face_distances = face_recognition.face_distance(
            self.known_encodings,
            face_encoding
        )

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return self.known_names[best_match_index]

        return "Unknown"

    # ---------------------------------------------------------
    # ➕ CALL AFTER ADDING NEW FACE (LIVE RELOAD)
    # ---------------------------------------------------------

    def reload(self):
        self.load_faces()