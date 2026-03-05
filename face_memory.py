import face_recognition
import os
import numpy as np

KNOWN_FACES_DIR = "known_faces"


class FaceMemory:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_faces()

    # -----------------------------
    # Load all saved faces
    # -----------------------------
    def load_faces(self):
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)

        for file in os.listdir(KNOWN_FACES_DIR):
            if file.endswith(".npy"):
                path = os.path.join(KNOWN_FACES_DIR, file)
                encoding = np.load(path)
                name = file.replace(".npy", "")

                self.known_encodings.append(encoding)
                self.known_names.append(name)

    # -----------------------------
    # Recognize face
    # -----------------------------
    def recognize(self, face_encoding):

        if len(self.known_encodings) == 0:
            return "Unknown"

        matches = face_recognition.compare_faces(
            self.known_encodings, face_encoding, tolerance=0.5
        )

        face_distances = face_recognition.face_distance(
            self.known_encodings, face_encoding
        )

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return self.known_names[best_match_index]

        return "Unknown"

    # -----------------------------
    # Save new face
    # -----------------------------
    def save_face(self, name, face_encoding):

        path = os.path.join(KNOWN_FACES_DIR, f"{name}.npy")

        np.save(path, face_encoding)

        self.known_encodings.append(face_encoding)
        self.known_names.append(name)

        print(f"{name} saved successfully.")

    # -----------------------------
    # Rename existing face
    # -----------------------------
    def rename_face(self, old_name, new_name):

        old_path = os.path.join(KNOWN_FACES_DIR, f"{old_name}.npy")
        new_path = os.path.join(KNOWN_FACES_DIR, f"{new_name}.npy")

        if os.path.exists(old_path):
            os.rename(old_path, new_path)

            index = self.known_names.index(old_name)
            self.known_names[index] = new_name

            print(f"{old_name} renamed to {new_name}")
        else:
            print("Face not found.")
