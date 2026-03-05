import json
import os
import hashlib

USERS_FILE = "gui/users.json"
SECRET_KEY = "CNA#123"


class AuthManager:

    def __init__(self):
        os.makedirs("gui", exist_ok=True)

        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, "w") as f:
                json.dump({}, f)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def load_users(self):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

        if not isinstance(users, dict):
            return {}

        sanitized = {}
        changed = False
        for user_id, pwd_hash in users.items():
            valid_user, _ = self.validate_user_id(user_id)
            if valid_user and isinstance(pwd_hash, str) and pwd_hash:
                sanitized[user_id.strip()] = pwd_hash
            else:
                changed = True

        if changed:
            self.save_users(sanitized)

        return sanitized

    def save_users(self, users):
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

    def validate_user_id(self, user_id):
        user_id = (user_id or "").strip()
        if not user_id:
            return False, "User ID is required"
        if len(user_id) < 3 or len(user_id) > 32:
            return False, "User ID must be 3-32 characters"
        return True, ""

    def validate_password(self, password):
        password = password or ""
        if len(password) < 8 or len(password) > 12:
            return False, "Password must be 8-12 characters"
        return True, ""

    def verify_secret_key(self, secret_key):
        return (secret_key or "").strip() == SECRET_KEY

    def create_user(self, user_id, password):
        user_id = (user_id or "").strip()
        valid_user, user_msg = self.validate_user_id(user_id)
        if not valid_user:
            return False, user_msg

        valid_pass, pass_msg = self.validate_password(password)
        if not valid_pass:
            return False, pass_msg

        users = self.load_users()

        if user_id in users:
            return False, "User already exists"

        users[user_id] = self.hash_password(password)
        self.save_users(users)

        return True, "Account created"

    def login(self, user_id, password):
        user_id = (user_id or "").strip()

        users = self.load_users()

        if user_id not in users:
            return False

        return users[user_id] == self.hash_password(password)

    def reset_password(self, user_id, new_password):
        user_id = (user_id or "").strip()
        valid_pass, pass_msg = self.validate_password(new_password)
        if not valid_pass:
            return False, pass_msg

        users = self.load_users()

        if user_id not in users:
            return False, "User not found"

        users[user_id] = self.hash_password(new_password)
        self.save_users(users)

        return True, "Password updated"
