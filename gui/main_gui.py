import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QStackedWidget, QDialog, QGraphicsBlurEffect,
    QFrame, QLabel, QGraphicsOpacityEffect, QSizePolicy
)
from PySide6.QtCore import QEvent, QEasingCurve, QPropertyAnimation, QRect, Qt, QTimer

from gui.utils import resource_path
from gui.login_dialog import LoginDialog
from gui.settings_manager import SettingsManager

# pages
from gui.ui.dashboard_page import DashboardPage
from gui.ui.attendance_page import AttendancePage
from gui.ui.model_page import ModelPage
from gui.ui.training_page import TrainingPage
from gui.ui.settings_page import SettingsPage
from gui.ui.about_page import AboutPage
from gui.ui.tracking_page import TrackingPage
from gui.ui.timetable_page import TimeTablePage
from gui.ui.database_page import DatabasePage
from gui.ui.behavior_page import BehaviorPage
from gui.ui.multi_camera_view_page import MultiCameraViewPage
from gui.ui.emotion_analytics import EmotionAnalyticsPage
from gui.ui.emotion_performance_page import EmotionPerformanceAnalyticsPage
from gui.ui.focus_monitoring_page import FocusModeMonitoringPage

# ⭐ TIMETABLE ENGINE
from features.engine.timetable_engine import TimeTableEngine


class OverlayDimmer(QWidget):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.setStyleSheet("background-color: rgba(0, 0, 0, 115);")
        self.opacity = QGraphicsOpacityEffect(self)
        self.opacity.setOpacity(0.0)
        self.setGraphicsEffect(self.opacity)
        self.fade_animation = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_animation.setDuration(180)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.hide()

    def fade_in(self):
        self.show()
        self.fade_animation.stop()
        self.fade_animation.setStartValue(self.opacity.opacity())
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()

    def fade_out(self):
        self.fade_animation.stop()
        try:
            self.fade_animation.finished.disconnect(self.hide)
        except (RuntimeError, TypeError):
            pass
        self.fade_animation.setStartValue(self.opacity.opacity())
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.hide)
        self.fade_animation.start()

    def mousePressEvent(self, event):
        if self.controller is not None:
            self.controller.close_overlay_sidebar()
        super().mousePressEvent(event)


class EdgeHoverZone(QWidget):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent;")

    def enterEvent(self, event):
        if self.controller is not None:
            self.controller.handle_edge_hover()
        super().enterEvent(event)


class MainWindow(QMainWindow):
    SIDEBAR_EXPANDED_WIDTH = 250
    SIDEBAR_COLLAPSED_WIDTH = 68
    SIDEBAR_HIDDEN_WIDTH = 0
    HIDDEN_MENU_RAIL_WIDTH = 96
    SIDEBAR_ANIMATION_MS = 240
    SMALL_WINDOW_WIDTH = 900
    WORKSPACE_PAGES = {"Live Tracking", "Emotion Analytics", "Focus Mode Monitoring"}

    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon(resource_path("gui/assets/branding/DrishtiAI_Logo.ico")))
        self.setWindowTitle("DrishtiAI 0.1")
        self.resize(1200, 700)
        self.setMinimumSize(1000, 620)
        self.settings_manager = SettingsManager()
        self.sidebar_settings = self.settings_manager.load()
        self.sidebar_mode = "expanded"
        self.last_standard_sidebar_mode = "expanded"
        self.overlay_active = False
        self.workspace_active = False
        self._responsive_forced_collapse = False
        self.nav_buttons = []
        self.nav_button_map = {}

        # ================= MAIN LAYOUT =================
        main_widget = QWidget()
        main_widget.setObjectName("smartRoot")
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.main_widget = main_widget
        self.main_layout = main_layout

        # ================= SIDEBAR =================
        self.sidebar_widget = QFrame(main_widget)
        self.sidebar_widget.setObjectName("smartSidebar")
        self.sidebar_widget.setMinimumWidth(self.SIDEBAR_EXPANDED_WIDTH)
        self.sidebar_widget.setMaximumWidth(self.SIDEBAR_EXPANDED_WIDTH)
        self.sidebar_widget.installEventFilter(self)
        self.sidebar_widget.setStyleSheet(
            """
            QFrame#smartSidebar {
                background-color:#151515;
                border-right:1px solid #2b2b2b;
            }
            QFrame#smartSidebar QPushButton {
                background-color:#1f1f1f;
                border:none;
                padding:10px;
                text-align:left;
                border-radius:6px;
            }
            QFrame#smartSidebar QPushButton:hover {
                background-color:#30343a;
                border-left:3px solid #4EA1FF;
            }
            QLabel#sidebarBrand {
                color:#f4f7fb;
                font-size:16px;
                font-weight:bold;
                padding:10px;
            }
            """
        )
        sidebar = QVBoxLayout(self.sidebar_widget)
        sidebar.setContentsMargins(8, 8, 8, 8)
        sidebar.setSpacing(6)

        sidebar_header = QHBoxLayout()
        self.sidebar_brand = QLabel("DrishtiAI")
        self.sidebar_brand.setObjectName("sidebarBrand")
        self.sidebar_toggle_btn = QPushButton("Menu")
        self.sidebar_toggle_btn.setToolTip("Collapse or expand sidebar")
        self.sidebar_toggle_btn.setFixedHeight(34)
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        self.sidebar_hide_btn = QPushButton("Hide")
        self.sidebar_hide_btn.setToolTip("Hide sidebar")
        self.sidebar_hide_btn.setFixedHeight(34)
        self.sidebar_hide_btn.clicked.connect(self.hide_sidebar)
        sidebar_header.addWidget(self.sidebar_brand, 1)
        sidebar_header.addWidget(self.sidebar_toggle_btn)
        sidebar_header.addWidget(self.sidebar_hide_btn)
        sidebar.addLayout(sidebar_header)

        btn_dashboard = QPushButton("Dashboard")
        btn_tracking = QPushButton("Live Tracking")
        btn_multi_cam = QPushButton("Multi Camera View")
        btn_timetable = QPushButton("Time-Table Editor")
        btn_behavior = QPushButton("Noise and Misbehavior")
        btn_emotion_analytics = QPushButton("Emotion Analytics")
        btn_emotion_performance = QPushButton("Emotion Performance Analytics")
        btn_focus_monitoring = QPushButton("Focus Mode Monitoring")
        btn_attendance = QPushButton("Attendance")
        btn_database = QPushButton("Database")
        btn_model = QPushButton("Model Performance")
        btn_training = QPushButton("Training")
        btn_settings = QPushButton("Settings")
        btn_about = QPushButton("About")

        buttons = [
            ("Dashboard", btn_dashboard),
            ("Live Tracking", btn_tracking),
            ("Multi Camera View", btn_multi_cam),
            ("Time-Table Editor", btn_timetable),
            ("Noise and Misbehavior", btn_behavior),
            ("Emotion Analytics", btn_emotion_analytics),
            ("Emotion Performance Analytics", btn_emotion_performance),
            ("Focus Mode Monitoring", btn_focus_monitoring),
            ("Attendance", btn_attendance),
            ("Database", btn_database),
            ("Model Performance", btn_model),
            ("Training", btn_training),
            ("Settings", btn_settings),
            ("About", btn_about),
        ]

        for label, btn in buttons:
            btn.setProperty("fullText", label)
            btn.setProperty("compactText", self._compact_label(label))
            btn.setMinimumHeight(40)
            sidebar.addWidget(btn)
            self.nav_buttons.append(btn)
            self.nav_button_map[label] = btn

        sidebar.addStretch()

        self.hidden_menu_rail = QFrame(main_widget)
        self.hidden_menu_rail.setObjectName("hiddenMenuRail")
        self.hidden_menu_rail.setFixedWidth(self.HIDDEN_MENU_RAIL_WIDTH)
        self.hidden_menu_rail.setStyleSheet(
            """
            QFrame#hiddenMenuRail {
                background-color:#121212;
                border-right:1px solid #24282f;
            }
            """
        )
        self.hidden_menu_rail.hide()

        self.content_shell = QFrame(main_widget)
        self.content_shell.setObjectName("contentShell")
        self.content_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout = QVBoxLayout(self.content_shell)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.workspace_bar = QFrame(self.content_shell)
        self.workspace_bar.setObjectName("workspaceBar")
        self.workspace_bar.setStyleSheet(
            "QFrame#workspaceBar { background:#101419; border-bottom:1px solid #252c36; }"
            "QPushButton { background:#1f1f1f; border:none; padding:8px; border-radius:5px; }"
            "QPushButton:hover { background:#333333; }"
            "QLabel { color:#cbd5e1; padding-left:10px; }"
        )
        workspace_row = QHBoxLayout(self.workspace_bar)
        workspace_row.setContentsMargins(8, 5, 8, 5)
        self.workspace_title = QLabel("")
        self.focus_workspace_btn = QPushButton("Focus Workspace")
        self.focus_workspace_btn.clicked.connect(self.toggle_workspace_mode)
        workspace_row.addWidget(self.workspace_title, 1)
        workspace_row.addWidget(self.focus_workspace_btn)

        # ================= STACKED PAGES =================
        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dashboard_page = DashboardPage()
        self.tracking_page = TrackingPage()
        self.multi_camera_page = MultiCameraViewPage()
        self.timetable_page = TimeTablePage()
        self.behavior_page = BehaviorPage()
        self.emotion_analytics_page = EmotionAnalyticsPage()
        self.emotion_performance_page = EmotionPerformanceAnalyticsPage()
        self.focus_monitoring_page = FocusModeMonitoringPage()
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
        self.pages.addWidget(self.emotion_performance_page)
        self.pages.addWidget(self.focus_monitoring_page)
        self.pages.addWidget(self.attendance_page)
        self.pages.addWidget(self.database_page)
        self.pages.addWidget(self.model_page)
        self.pages.addWidget(self.training_page)
        self.pages.addWidget(self.settings_page)
        self.pages.addWidget(self.about_page)
        self._normalize_page_sizing()
        content_layout.addWidget(self.workspace_bar)
        content_layout.addWidget(self.pages, 1)

        # Motion analytics remains off by default to reduce load.
        self.emotion_analytics_page.set_runtime_enabled(False)

        # ================= CONNECTIONS =================
        self.route_map = {
            "Dashboard": self.dashboard_page,
            "Live Tracking": self.tracking_page,
            "Multi Camera View": self.multi_camera_page,
            "Time-Table Editor": self.timetable_page,
            "Noise and Misbehavior": self.behavior_page,
            "Emotion Analytics": self.emotion_analytics_page,
            "Emotion Performance Analytics": self.emotion_performance_page,
            "Focus Mode Monitoring": self.focus_monitoring_page,
            "Attendance": self.attendance_page,
            "Database": self.database_page,
            "Model Performance": self.model_page,
            "Training": self.training_page,
            "Settings": self.settings_page,
            "About": self.about_page,
        }
        for label, button in self.nav_button_map.items():
            button.clicked.connect(lambda checked=False, item=label: self.navigate_to(item))
        self.dashboard_page.motion_analytics_toggled.connect(
            self.emotion_analytics_page.set_runtime_enabled
        )
        self.settings_page.settings_saved.connect(self.apply_sidebar_preferences)
        self.pages.currentChanged.connect(self.on_page_changed)

        # ================= ADD TO MAIN LAYOUT =================
        main_layout.addWidget(self.sidebar_widget)
        main_layout.addWidget(self.hidden_menu_rail)
        main_layout.addWidget(self.content_shell, 1)

        self.setCentralWidget(main_widget)
        self.dim_overlay = OverlayDimmer(main_widget, controller=self)
        self.overlay_sidebar = None
        self.edge_hover_zone = EdgeHoverZone(main_widget, controller=self)
        self.edge_hover_zone.hide()
        self.fab_button = QPushButton("Menu", main_widget)
        self.fab_button.setObjectName("sidebarFab")
        self.fab_button.setFixedSize(82, 34)
        self.fab_button.setStyleSheet(
            "QPushButton#sidebarFab {"
            " background:#1f1f1f;"
            " color:#f4f7fb;"
            " border:1px solid #343a40;"
            " border-radius:17px;"
            " font-weight:bold;"
            " text-align:center;"
            "}"
            "QPushButton#sidebarFab:hover {"
            " background:#2b3036;"
            " border:1px solid #4EA1FF;"
            " color:#ffffff;"
            "}"
        )
        self.fab_button.clicked.connect(self.open_overlay_sidebar)
        self.fab_button.hide()
        self.sidebar_animation = QPropertyAnimation(self.sidebar_widget, b"maximumWidth", self)
        self.sidebar_animation.setDuration(self.SIDEBAR_ANIMATION_MS)
        self.sidebar_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.sidebar_animation.finished.connect(self.refresh_current_page)
        self.apply_sidebar_preferences(self.sidebar_settings)
        self.on_page_changed(self.pages.currentIndex())

        # ================= TIMETABLE ENGINE =================
        self.timetable_engine = TimeTableEngine()

        self.clock = QTimer()
        self.clock.timeout.connect(self.check_timetable)
        self.clock.start(10000)   # every 10 seconds

        # ================= LOGIN BLUR =================
        self.apply_blur()
        self.show_login()

    # ================= SMART SIDEBAR =================
    def navigate_to(self, label):
        page = self.route_map.get(label)
        if page is None:
            return
        self.pages.setCurrentWidget(page)
        if self.overlay_active:
            self.close_overlay_sidebar()

    def toggle_sidebar(self):
        if self.sidebar_mode == "expanded":
            self.set_sidebar_mode("collapsed")
        else:
            self.set_sidebar_mode("expanded")

    def hide_sidebar(self):
        self.set_sidebar_mode("hidden")

    def set_sidebar_mode(self, mode, animated=True, hover=False):
        if mode not in {"expanded", "collapsed", "hidden"}:
            return
        if self.workspace_active and mode != "hidden" and not hover:
            self.workspace_active = False
            self.focus_workspace_btn.setText("Focus Workspace")

        self.sidebar_mode = "hover-expanded" if hover and mode == "expanded" else mode
        if mode in {"expanded", "collapsed"} and not hover:
            self.last_standard_sidebar_mode = mode

        target = {
            "expanded": self.SIDEBAR_EXPANDED_WIDTH,
            "collapsed": self.SIDEBAR_COLLAPSED_WIDTH,
            "hidden": self.SIDEBAR_HIDDEN_WIDTH,
        }[mode]

        self.sidebar_widget.show()
        self.sidebar_widget.setMinimumWidth(0)
        self.sidebar_animation.stop()
        if animated:
            self.sidebar_animation.setStartValue(self.sidebar_widget.width())
            self.sidebar_animation.setEndValue(target)
            self.sidebar_animation.start()
        else:
            self.sidebar_widget.setMaximumWidth(target)
            self.sidebar_widget.setMinimumWidth(target)
            self.refresh_current_page()

        self._apply_sidebar_text_mode(mode)
        self._update_sidebar_float_controls(mode)
        QTimer.singleShot(self.SIDEBAR_ANIMATION_MS + 20, lambda: self._lock_sidebar_width(target))

    def _lock_sidebar_width(self, width):
        if self.sidebar_animation.state() == QPropertyAnimation.Running:
            return
        self.sidebar_widget.setMinimumWidth(width)
        self.sidebar_widget.setMaximumWidth(width)
        self.refresh_current_page()

    def _apply_sidebar_text_mode(self, mode):
        compact = mode == "collapsed" or bool(self.sidebar_settings.get("sidebar_compact_icons", False))
        self.sidebar_brand.setVisible(mode != "collapsed")
        self.sidebar_hide_btn.setVisible(mode != "collapsed")
        for button in self.nav_buttons:
            full_text = button.property("fullText")
            compact_text = button.property("compactText")
            button.setText(compact_text if compact else full_text)
            button.setToolTip(full_text)

    def _compact_label(self, label):
        shortcuts = {
            "Dashboard": "DB",
            "Live Tracking": "LT",
            "Multi Camera View": "MC",
            "Time-Table Editor": "TT",
            "Noise and Misbehavior": "NM",
            "Emotion Analytics": "EA",
            "Emotion Performance Analytics": "EP",
            "Focus Mode Monitoring": "FM",
            "Attendance": "AT",
            "Database": "DS",
            "Model Performance": "MP",
            "Training": "TR",
            "Settings": "ST",
            "About": "AB",
        }
        return shortcuts.get(label, label[:2].upper())

    def _normalize_page_sizing(self):
        for page in (
            self.dashboard_page,
            self.tracking_page,
            self.multi_camera_page,
            self.timetable_page,
            self.behavior_page,
            self.emotion_analytics_page,
            self.emotion_performance_page,
            self.focus_monitoring_page,
            self.attendance_page,
            self.database_page,
            self.model_page,
            self.training_page,
            self.settings_page,
            self.about_page,
        ):
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            page.setMinimumSize(0, 0)

    def _update_sidebar_float_controls(self, mode):
        hidden = mode == "hidden"
        self.hidden_menu_rail.setVisible(hidden)
        self.fab_button.setVisible(hidden)
        self.content_shell.setContentsMargins(0, 0, 0, 0)
        self.edge_hover_zone.setVisible(
            hidden and bool(self.sidebar_settings.get("sidebar_hover_expand", True))
        )
        self._position_floating_controls()

    def open_overlay_sidebar(self):
        if not bool(self.sidebar_settings.get("sidebar_overlay_navigation", True)):
            self.set_sidebar_mode("expanded")
            return

        self.overlay_active = True
        self.dim_overlay.setGeometry(self.main_widget.rect())
        self.dim_overlay.fade_in()
        self.dim_overlay.raise_()

        if self.overlay_sidebar is None:
            self.overlay_sidebar = self._build_overlay_sidebar()
        self.overlay_sidebar.setGeometry(-self.SIDEBAR_EXPANDED_WIDTH, 0, self.SIDEBAR_EXPANDED_WIDTH, self.main_widget.height())
        self.overlay_sidebar.show()
        self.overlay_sidebar.raise_()

        animation = QPropertyAnimation(self.overlay_sidebar, b"geometry", self)
        animation.setDuration(self.SIDEBAR_ANIMATION_MS)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.setStartValue(QRect(-self.SIDEBAR_EXPANDED_WIDTH, 0, self.SIDEBAR_EXPANDED_WIDTH, self.main_widget.height()))
        animation.setEndValue(QRect(0, 0, self.SIDEBAR_EXPANDED_WIDTH, self.main_widget.height()))
        self.overlay_animation = animation
        animation.start()

    def close_overlay_sidebar(self):
        if not self.overlay_active:
            return
        self.overlay_active = False
        if self.overlay_sidebar is None:
            self.dim_overlay.hide()
            return

        animation = QPropertyAnimation(self.overlay_sidebar, b"geometry", self)
        animation.setDuration(self.SIDEBAR_ANIMATION_MS)
        animation.setEasingCurve(QEasingCurve.InCubic)
        animation.setStartValue(self.overlay_sidebar.geometry())
        animation.setEndValue(QRect(-self.SIDEBAR_EXPANDED_WIDTH, 0, self.SIDEBAR_EXPANDED_WIDTH, self.main_widget.height()))
        animation.finished.connect(self.overlay_sidebar.hide)
        animation.finished.connect(self.dim_overlay.fade_out)
        self.overlay_animation = animation
        animation.start()

    def _build_overlay_sidebar(self):
        panel = QFrame(self.main_widget)
        panel.setObjectName("overlaySidebar")
        panel.setStyleSheet(
            """
            QFrame#overlaySidebar {
                background-color:#151515;
                border-right:1px solid #2b2b2b;
            }
            QFrame#overlaySidebar QPushButton {
                background-color:#1f1f1f;
                border:none;
                padding:10px;
                text-align:left;
                border-radius:6px;
            }
            QFrame#overlaySidebar QPushButton:hover {
                background-color:#30343a;
                border-left:3px solid #4EA1FF;
            }
            QLabel { color:#f4f7fb; font-size:16px; font-weight:bold; padding:10px; }
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        header = QHBoxLayout()
        header.addWidget(QLabel("DrishtiAI"), 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close_overlay_sidebar)
        header.addWidget(close_btn)
        layout.addLayout(header)
        for label in self.route_map.keys():
            button = QPushButton(label)
            button.setMinimumHeight(40)
            button.clicked.connect(lambda checked=False, item=label: self.navigate_to(item))
            layout.addWidget(button)
        layout.addStretch()
        panel.hide()
        return panel

    def handle_edge_hover(self):
        if not bool(self.sidebar_settings.get("sidebar_hover_expand", True)):
            return
        if self.sidebar_mode == "hidden" and bool(self.sidebar_settings.get("sidebar_overlay_navigation", True)):
            self.open_overlay_sidebar()
        elif self.sidebar_mode in {"collapsed", "hidden"}:
            self.set_sidebar_mode("expanded", hover=True)

    def eventFilter(self, watched, event):
        if watched == self.sidebar_widget:
            if event.type() == QEvent.Enter and self.sidebar_mode == "collapsed":
                if bool(self.sidebar_settings.get("sidebar_hover_expand", True)):
                    self.set_sidebar_mode("expanded", hover=True)
            elif event.type() == QEvent.Leave and self.sidebar_mode == "hover-expanded":
                self.set_sidebar_mode("collapsed")
        return super().eventFilter(watched, event)

    def apply_sidebar_preferences(self, settings):
        self.sidebar_settings = settings or self.settings_manager.load()
        if bool(self.sidebar_settings.get("sidebar_fullscreen_workspace", False)):
            self.workspace_active = True
            self.focus_workspace_btn.setText("Exit Workspace")
            self.set_sidebar_mode("hidden", animated=False)
        elif bool(self.sidebar_settings.get("sidebar_auto_hide", False)):
            self.set_sidebar_mode("hidden", animated=False)
        else:
            self.set_sidebar_mode(self.last_standard_sidebar_mode, animated=False)

    def toggle_workspace_mode(self):
        self.workspace_active = not self.workspace_active
        self.focus_workspace_btn.setText("Exit Workspace" if self.workspace_active else "Focus Workspace")
        if self.workspace_active:
            self.set_sidebar_mode("hidden")
        else:
            self.set_sidebar_mode(self.last_standard_sidebar_mode)

    def on_page_changed(self, index):
        label = self._current_page_label()
        workspace_allowed = label in self.WORKSPACE_PAGES
        self.workspace_bar.setVisible(workspace_allowed)
        self.workspace_title.setText(label)
        if self.workspace_active and not workspace_allowed:
            self.workspace_active = False
            self.focus_workspace_btn.setText("Focus Workspace")
            if not bool(self.sidebar_settings.get("sidebar_auto_hide", False)):
                self.set_sidebar_mode(self.last_standard_sidebar_mode)
        self.refresh_current_page()

    def _current_page_label(self):
        current = self.pages.currentWidget()
        for label, page in self.route_map.items():
            if page == current:
                return label
        return ""

    def refresh_current_page(self):
        page = self.pages.currentWidget()
        if page is None:
            return
        page.updateGeometry()
        page.update()
        for canvas in page.findChildren(QWidget):
            if hasattr(canvas, "draw_idle"):
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

    def _position_floating_controls(self):
        if not hasattr(self, "fab_button"):
            return
        margin = 16
        if not self.fab_button.isHidden():
            rail_width = self.hidden_menu_rail.width() or self.HIDDEN_MENU_RAIL_WIDTH
            x = max(6, int((rail_width - self.fab_button.width()) / 2))
            y = margin
            self.fab_button.move(x, y)
        if hasattr(self, "edge_hover_zone"):
            self.edge_hover_zone.setGeometry(0, 0, 4, self.main_widget.height())
            self.edge_hover_zone.raise_()
        self.fab_button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "dim_overlay"):
            self.dim_overlay.setGeometry(self.main_widget.rect())
        if self.overlay_sidebar is not None and self.overlay_sidebar.isVisible():
            self.overlay_sidebar.setGeometry(0, 0, self.SIDEBAR_EXPANDED_WIDTH, self.main_widget.height())
        self._position_floating_controls()
        self._apply_responsive_sidebar()
        QTimer.singleShot(80, self.refresh_current_page)

    def _apply_responsive_sidebar(self):
        if self.workspace_active or self.sidebar_mode in {"hidden", "hover-expanded"}:
            return
        if self.width() < self.SMALL_WINDOW_WIDTH and self.sidebar_mode == "expanded":
            self._responsive_forced_collapse = True
            self.set_sidebar_mode("collapsed")
        elif self.width() >= self.SMALL_WINDOW_WIDTH and self._responsive_forced_collapse:
            self._responsive_forced_collapse = False
            if not bool(self.sidebar_settings.get("sidebar_auto_hide", False)):
                self.set_sidebar_mode("expanded")

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
