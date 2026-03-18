import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from gui.emotion_model_runtime import (
    get_dataset_split_dir,
    get_output_units,
    get_preferred_model_path,
    normalize_class_names,
)


class ModelMetrics:
    def __init__(self, model_path, test_dir=None):
        self.model_path = model_path
        self.test_dir = test_dir

        self.model = None
        self.X = None
        self.y = None
        self.class_names = []
        self.output_units = None

        self.load_model()
        self.load_test_data()

    def load_model(self):
        self.model_path = get_preferred_model_path(self.model_path)

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False)
            self.output_units = get_output_units(self.model)
            print("Model loaded for evaluation")
        else:
            print("Model file not found:", self.model_path)

    def load_test_data(self):
        if not self.test_dir:
            self.test_dir = get_dataset_split_dir("test", output_units=self.output_units)

        if not os.path.exists(self.test_dir):
            print("Test directory not found:", self.test_dir)
            return

        datagen = ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_directory(
            self.test_dir,
            target_size=(96, 96),
            batch_size=32,
            class_mode="categorical",
            shuffle=False,
        )

        self.X = generator
        self.y = generator.classes
        self.class_names = normalize_class_names(generator.class_indices.keys())

        print("Test dataset loaded")

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model not loaded. Check model path.")

        if self.X is None:
            raise ValueError("Test data not loaded.")

        preds = self.model.predict(self.X, verbose=0)
        if preds.shape[1] != len(self.class_names):
            raise ValueError(
                f"Model outputs {preds.shape[1]} classes but test dataset has "
                f"{len(self.class_names)} folders at {self.test_dir}."
            )

        y_pred = np.argmax(preds, axis=1)

        acc = accuracy_score(self.y, y_pred)
        prec = precision_score(self.y, y_pred, average="weighted", zero_division=0)
        rec = recall_score(self.y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(self.y, y_pred)

        report = classification_report(
            self.y,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return acc, prec, rec, f1, cm, report
