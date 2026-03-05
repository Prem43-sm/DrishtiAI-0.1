import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelMetrics:

    def __init__(self, model_path, test_dir):

        self.model_path = model_path
        self.test_dir = test_dir

        self.model = None
        self.X = None
        self.y = None

        self.load_model()
        self.load_test_data()

    # --------------------------------------------------

    def load_model(self):

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False)
            print("✅ Model loaded for evaluation")
        else:
            print("❌ Model file not found:", self.model_path)

    # --------------------------------------------------

    def load_test_data(self):

        if not os.path.exists(self.test_dir):
            print("❌ Test directory not found:", self.test_dir)
            return

        datagen = ImageDataGenerator(rescale=1.0 / 255)

        generator = datagen.flow_from_directory(
            self.test_dir,
            target_size=(96, 96),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        self.X = generator
        self.y = generator.classes
        self.class_names = list(generator.class_indices.keys())

        print("✅ Test dataset loaded")

    # --------------------------------------------------

    def evaluate(self):

        if self.model is None:
            raise ValueError("Model not loaded. Check model path.")

        if self.X is None:
            raise ValueError("Test data not loaded.")

        # Predict
        preds = self.model.predict(self.X, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        # Metrics
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
            zero_division=0
        )

        return acc, prec, rec, f1, cm, report