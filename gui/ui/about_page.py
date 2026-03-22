from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.utils import resource_path


class AboutPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        page_title = QLabel("About")
        page_title.setAlignment(Qt.AlignCenter)
        page_title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(page_title)

        college = QLabel(
            "Govt. E. Raghvendra Rao P.G. Science College\nSarkanda Bilaspur"
        )
        college.setAlignment(Qt.AlignCenter)
        college.setStyleSheet("font-size:20px; font-weight:bold; color:#00E5FF;")

        logo = QLabel()
        logo_path = resource_path("gui/assets/branding/logo.png")
        pixmap = QPixmap(logo_path)

        if pixmap.isNull():
            logo.setText("Logo not found")
        else:
            logo.setPixmap(
                pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        logo.setAlignment(Qt.AlignCenter)

        title = QLabel("DrishtiAI 0.1\nIntelligent Attendance & Surveillance System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:16px; font-weight:bold;")

        dev = QLabel("Developed by:\nPremnarayan Chandra\nMSc IT - 4th Semester")
        dev.setAlignment(Qt.AlignCenter)
        dev.setStyleSheet("font-size:13px;")

        hod = QLabel(" Guided by: \nHOD: Dr Kajal Kiran Gulhare Ma'am ")
        hod.setAlignment(Qt.AlignCenter)

        guide = QLabel(" Guide: Sumati Pathak Ma'am ")
        guide.setAlignment(Qt.AlignCenter)

        layout.addStretch()
        layout.addWidget(college)
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(dev)
        layout.addWidget(hod)
        layout.addWidget(guide)
        layout.addStretch()

        self.setLayout(layout)
