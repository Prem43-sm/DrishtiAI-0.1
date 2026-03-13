import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QStackedWidget, QDialog, QGraphicsBlurEffect
)
from PySide6.QtCore import QTimer

from utils import resource_path
from login_dialog import LoginDialog

# pages
from ui.dashboard_page import DashboardPage
from ui.attendance_page import AttendancePage
from ui.model_page import ModelPage
from ui.training_page import TrainingPage
from ui.settings_page import SettingsPage
from ui.about_page import AboutPage
from ui.tracking_page import TrackingPage
from ui.timetable_page import TimeTablePage
from ui.database_page import DatabasePage
from ui.behavior_page import BehaviorPage
from ui.multi_camera_view_page import MultiCameraViewPage
from ui.emotion_analytics import EmotionAnalyticsPage

# ⭐ TIMETABLE ENGINE
from features.engine.timetable_engine import TimeTableEngine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon(resource_path("gui/assets/branding/DrishtiAI_Logo.ico")))
        self.setWindowTitle("DrishtiAI 0.1")
        self.resize(1200, 700)

        # ================= MAIN LAYOUT =================
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # ================= SIDEBAR =================
        sidebar = QVBoxLayout()

        btn_dashboard = QPushButton("Dashboard")
        btn_tracking = QPushButton("Live Tracking")
        btn_multi_cam = QPushButton("Multi Camera View")
        btn_timetable = QPushButton("Time-Table Editor")
        btn_behavior = QPushButton("Noise and Misbehavior")
        btn_emotion_analytics = QPushButton("Emotion Analytics")
        btn_attendance = QPushButton("Attendance")
        btn_database = QPushButton("Database")
        btn_model = QPushButton("Model Performance Score")
        btn_training = QPushButton("Training")
        btn_settings = QPushButton("Settings")
        btn_about = QPushButton("About")

        buttons = [
            btn_dashboard,
            btn_tracking,
            btn_multi_cam,
            btn_timetable,
            btn_behavior,
            btn_emotion_analytics,
            btn_attendance,
            btn_database,
            btn_model,
            btn_training,
            btn_settings,
            btn_about
        ]

        for btn in buttons:
            btn.setMinimumHeight(40)
            sidebar.addWidget(btn)

        sidebar.addStretch()

        # ================= STACKED PAGES =================
        self.pages = QStackedWidget()

        self.dashboard_page = DashboardPage()
        self.tracking_page = TrackingPage()
        self.multi_camera_page = MultiCameraViewPage()
        self.timetable_page = TimeTablePage()
        self.behavior_page = BehaviorPage()
        self.emotion_analytics_page = EmotionAnalyticsPage()
        self.attendance_page = AttendancePage()
        self.database_page = DatabasePage()
        self.model_page = ModelPage()
        self.training_page = TrainingPage()
        self.settings_page = SettingsPage()
        self.about_page = AboutPage()

        self.pages.addWidget(self.dashboard_page)
        self.pages.addWidget(self.tracking_page)
        self.pages.addWidget(self.multi_camera_page)
        self.pages.addWidget(self.timetable_page)
        self.pages.addWidget(self.behavior_page)
        self.pages.addWidget(self.emotion_analytics_page)
        self.pages.addWidget(self.attendance_page)
        self.pages.addWidget(self.database_page)
        self.pages.addWidget(self.model_page)
        self.pages.addWidget(self.training_page)
        self.pages.addWidget(self.settings_page)
        self.pages.addWidget(self.about_page)

        # Motion analytics remains off by default to reduce load.
        self.emotion_analytics_page.set_runtime_enabled(False)

        # ================= CONNECTIONS =================
        btn_dashboard.clicked.connect(lambda: self.pages.setCurrentWidget(self.dashboard_page))
        btn_tracking.clicked.connect(lambda: self.pages.setCurrentWidget(self.tracking_page))
        btn_multi_cam.clicked.connect(lambda: self.pages.setCurrentWidget(self.multi_camera_page))
        btn_timetable.clicked.connect(lambda: self.pages.setCurrentWidget(self.timetable_page))
        btn_behavior.clicked.connect(lambda: self.pages.setCurrentWidget(self.behavior_page))
        btn_emotion_analytics.clicked.connect(lambda: self.pages.setCurrentWidget(self.emotion_analytics_page))
        btn_attendance.clicked.connect(lambda: self.pages.setCurrentWidget(self.attendance_page))
        btn_database.clicked.connect(lambda: self.pages.setCurrentWidget(self.database_page))
        btn_model.clicked.connect(lambda: self.pages.setCurrentWidget(self.model_page))
        btn_training.clicked.connect(lambda: self.pages.setCurrentWidget(self.training_page))
        btn_settings.clicked.connect(lambda: self.pages.setCurrentWidget(self.settings_page))
        btn_about.clicked.connect(lambda: self.pages.setCurrentWidget(self.about_page))
        self.dashboard_page.motion_analytics_toggled.connect(
            self.emotion_analytics_page.set_runtime_enabled
        )

        # ================= ADD TO MAIN LAYOUT =================
        main_layout.addLayout(sidebar, 1)
        main_layout.addWidget(self.pages, 4)

        self.setCentralWidget(main_widget)

        # ================= TIMETABLE ENGINE =================
        self.timetable_engine = TimeTableEngine()

        self.clock = QTimer()
        self.clock.timeout.connect(self.check_timetable)
        self.clock.start(10000)   # every 10 seconds

        # ================= LOGIN BLUR =================
        self.apply_blur()
        self.show_login()

    # ================= TIMETABLE CHECK =================
    def check_timetable(self):

        active = self.timetable_engine.check_current_slot()

        if active:
            class_name = self.timetable_engine.get_active_class()
            period = self.timetable_engine.get_active_period()

            print("ACTIVE:", class_name, "Period:", period)

            # 🔥 tell attendance page
            self.attendance_page.set_active_class(class_name, period)

        else:
            self.attendance_page.stop_auto_attendance()

    # ================= BLUR =================
    def apply_blur(self):
        self.blur = QGraphicsBlurEffect()
        self.blur.setBlurRadius(15)
        self.centralWidget().setGraphicsEffect(self.blur)

    def remove_blur(self):
        self.centralWidget().setGraphicsEffect(None)

    # ================= LOGIN =================
    def show_login(self):
        login = LoginDialog()
        if login.exec() == QDialog.Accepted:
            self.remove_blur()
        else:
            self.close()


# ================= APP START =================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #121212;
            color: white;
            font-size: 14px;
        }
        QPushButton {
            background-color: #1f1f1f;
            border: none;
            padding: 10px;
            text-align: left;
        }
        QPushButton:hover {
            background-color: #333333;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
