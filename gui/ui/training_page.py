import os
import tensorflow as tf

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QSpinBox,
    QFileDialog, QGroupBox, QFormLayout, QMessageBox,
    QToolButton
)

from PySide6.QtCore import QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from gui.emotion_model_runtime import DEFAULT_MODEL_PATH, get_dataset_split_dir


# ==============================
# 🧵 TRAINING THREAD
# ==============================
class TrainingWorker(QThread):

    progress = Signal(int, float, float, float, float)
    finished = Signal()

    def __init__(self, img, batch, epochs, save_path):
        super().__init__()
        self.img = img
        self.batch = batch
        self.epochs = epochs
        self.save_path = save_path

    def run(self):

        train_dir = get_dataset_split_dir("train")
        val_dir = get_dataset_split_dir("val")

        train_gen = ImageDataGenerator(rescale=1./255)
        val_gen = ImageDataGenerator(rescale=1./255)

        train = train_gen.flow_from_directory(
            train_dir,
            target_size=(self.img, self.img),
            batch_size=self.batch,
            class_mode="categorical"
        )

        val = val_gen.flow_from_directory(
            val_dir,
            target_size=(self.img, self.img),
            batch_size=self.batch,
            class_mode="categorical"
        )

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu',
                                   input_shape=(self.img, self.img, 3)),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(train.num_classes, activation='softmax')
        ])

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        checkpoint = ModelCheckpoint(
            self.save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )

        class ProgressCallback(Callback):
            def on_epoch_end(cb, epoch, logs=None):
                self.progress.emit(
                    epoch + 1,
                    logs["accuracy"],
                    logs["loss"],
                    logs["val_accuracy"],
                    logs["val_loss"]
                )

        model.fit(
            train,
            validation_data=val,
            epochs=self.epochs,
            callbacks=[checkpoint, ProgressCallback()],
            verbose=0
        )

        self.finished.emit()


# ==============================
# 🖥️ UI PAGE
# ==============================
class TrainingPage(QWidget):

    def __init__(self):
        super().__init__()

        self.worker = None
        layout = QVBoxLayout()

        # 🔷 TITLE + HELP BUTTON
        title_layout = QHBoxLayout()

        title = QLabel("Training")
        title.setStyleSheet("font-size:18px; font-weight:bold;")

        help_btn = QToolButton()
        help_btn.setText("❓")
        help_btn.clicked.connect(self.show_training_help)

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(help_btn)

        layout.addLayout(title_layout)

        # =====================
        # ⚙️ SETTINGS BOX
        # =====================
        box = QGroupBox("Training Settings")
        form = QFormLayout()

        self.img_size = QSpinBox()
        self.img_size.setValue(96)
        self.img_size.setToolTip("Recommended: 96 for speed & accuracy")

        self.batch_size = QSpinBox()
        self.batch_size.setValue(64)
        self.batch_size.setToolTip("Recommended: 64 for RTX 4050")

        self.epochs = QSpinBox()
        self.epochs.setValue(35)
        self.epochs.setToolTip("5=test | 15–25=normal | 35+=final")

        self.save_btn = QPushButton("Select Model Save Path")
        self.save_btn.clicked.connect(self.select_path)

        self.save_label = QLabel(DEFAULT_MODEL_PATH)
        self.save_path = DEFAULT_MODEL_PATH

        form.addRow("Image Size:", self.img_size)
        form.addRow("Batch Size:", self.batch_size)
        form.addRow("Epochs:", self.epochs)
        form.addRow(self.save_btn, self.save_label)

        box.setLayout(form)
        layout.addWidget(box)

        # ▶ START BUTTON
        self.start_btn = QPushButton("Start Training")
        layout.addWidget(self.start_btn)

        # 📊 PROGRESS
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # 📈 GRAPH
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_training)

    # =====================
    # ❓ HELP POPUP
    # =====================
    def show_training_help(self):

        QMessageBox.information(
            self,
            "Training Parameters Guide",

            "Image Size:\n"
            "96 → Fast & recommended\n"
            "128 → Higher accuracy but slower\n\n"

            "Batch Size:\n"
            "32 → For low GPU memory\n"
            "64 → Best for RTX 3050/4050 ✅\n"
            "128 → Needs high VRAM\n\n"

            "Epochs:\n"
            "5 → Testing\n"
            "15–25 → Normal training\n"
            "35+ → Final training\n"
        )

    # =====================
    # 📁 SELECT SAVE PATH
    # =====================
    def select_path(self):

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "H5 File (*.h5)"
        )

        if path:
            self.save_path = path
            self.save_label.setText(path)

    # =====================
    # ▶ START TRAINING
    # =====================
    def start_training(self):

        self.progress.setValue(0)
        self.ax.clear()
        self.canvas.draw()

        self.worker = TrainingWorker(
            self.img_size.value(),
            self.batch_size.value(),
            self.epochs.value(),
            self.save_path
        )

        self.worker.progress.connect(self.update_graph)
        self.worker.finished.connect(self.training_done)

        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []

        self.start_btn.setEnabled(False)
        self.worker.start()

    # =====================
    # 📈 UPDATE GRAPH
    # =====================
    def update_graph(self, epoch, acc, loss, val_acc, val_loss):

        percent = int((epoch / self.epochs.value()) * 100)
        self.progress.setValue(percent)

        self.acc.append(acc)
        self.loss.append(loss)
        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss)

        self.ax.clear()
        self.ax.plot(self.acc, label="Train Acc")
        self.ax.plot(self.val_acc, label="Val Acc")
        self.ax.plot(self.loss, label="Train Loss")
        self.ax.plot(self.val_loss, label="Val Loss")
        self.ax.legend()

        self.canvas.draw()

    def training_done(self):

        self.start_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Training Complete ✅",
            f"Model saved at:\n{self.save_path}"
        )
