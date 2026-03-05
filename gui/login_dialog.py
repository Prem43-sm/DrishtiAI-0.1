from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QHBoxLayout,
    QStackedWidget,
    QWidget,
    QInputDialog,
)

from auth_manager import AuthManager


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.auth = AuthManager()

        self.setWindowTitle("Login Required")
        self.setFixedWidth(380)

        root = QVBoxLayout()
        self.pages = QStackedWidget()
        root.addWidget(self.pages)
        self.setLayout(root)

        self.choice_page = self._build_choice_page()
        self.login_page = self._build_login_page()
        self.create_page = self._build_create_page()

        self.pages.addWidget(self.choice_page)
        self.pages.addWidget(self.login_page)
        self.pages.addWidget(self.create_page)
        self.pages.setCurrentWidget(self.choice_page)

    def _build_choice_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Select Option"))
        create_btn = QPushButton("1. Create Account")
        login_btn = QPushButton("2. Log In")
        layout.addWidget(create_btn)
        layout.addWidget(login_btn)

        create_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.create_page))
        login_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.login_page))
        return page

    def _build_login_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("User ID"))
        self.login_user = QLineEdit()
        layout.addWidget(self.login_user)

        layout.addWidget(QLabel("Password"))
        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.login_password)

        login_btn = QPushButton("Login")
        forgot_btn = QPushButton("Forgot Password")
        back_btn = QPushButton("Back")

        layout.addWidget(login_btn)

        row = QHBoxLayout()
        row.addWidget(forgot_btn)
        row.addWidget(back_btn)
        layout.addLayout(row)

        login_btn.clicked.connect(self.handle_login)
        forgot_btn.clicked.connect(self.reset_password)
        back_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.choice_page))
        return page

    def _build_create_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Secret Key"))
        self.create_secret = QLineEdit()
        self.create_secret.setEchoMode(QLineEdit.Password)
        self.create_secret.setPlaceholderText("Enter key")
        layout.addWidget(self.create_secret)

        layout.addWidget(QLabel("New User ID"))
        self.create_user = QLineEdit()
        layout.addWidget(self.create_user)

        layout.addWidget(QLabel("New Password"))
        self.create_password = QLineEdit()
        self.create_password.setEchoMode(QLineEdit.Password)
        self.create_password.setPlaceholderText("8-12 characters")
        layout.addWidget(self.create_password)

        create_btn = QPushButton("Create Account")
        back_btn = QPushButton("Back")

        layout.addWidget(create_btn)
        layout.addWidget(back_btn)

        create_btn.clicked.connect(self.create_account)
        back_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.choice_page))
        return page

    def handle_login(self):
        if self.auth.login(self.login_user.text(), self.login_password.text()):
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid user ID or password")

    def create_account(self):
        if not self.auth.verify_secret_key(self.create_secret.text()):
            QMessageBox.warning(self, "Error", "Invalid secret key")
            return

        ok, msg = self.auth.create_user(
            self.create_user.text(),
            self.create_password.text(),
        )

        if ok:
            QMessageBox.information(self, "Success", msg)
            self.login_user.setText(self.create_user.text().strip())
            self.login_password.clear()
            self.pages.setCurrentWidget(self.login_page)
        else:
            QMessageBox.warning(self, "Error", msg)

    def reset_password(self):
        user_id = self.login_user.text().strip()
        if not user_id:
            QMessageBox.warning(self, "Error", "Enter User ID first")
            return

        secret_key, key_ok = QInputDialog.getText(
            self,
            "Secret Key Required",
            "Enter secret key to reset password:",
            QLineEdit.Password,
        )
        if not key_ok:
            return
        if not self.auth.verify_secret_key(secret_key):
            QMessageBox.warning(self, "Error", "Invalid secret key")
            return

        new_password, pass_ok = QInputDialog.getText(
            self,
            "Reset Password",
            "Enter new password (8-12 characters):",
            QLineEdit.Password,
        )
        if not pass_ok:
            return

        ok, msg = self.auth.reset_password(user_id, new_password)
        if ok:
            QMessageBox.information(self, "Done", msg)
            self.login_password.clear()
        else:
            QMessageBox.warning(self, "Error", msg)
