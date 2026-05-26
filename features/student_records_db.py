from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.project_paths import ANALYTICS_DB_FILE, ensure_runtime_layout


ADMIN_ROLES = {"admin", "hod"}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_admin_role(role: str | None = None) -> bool:
    return str(role or "admin").strip().lower() in ADMIN_ROLES


def initialize_student_records_database() -> None:
    ensure_runtime_layout()
    with sqlite3.connect(str(ANALYTICS_DB_FILE)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                academic_year TEXT NOT NULL,
                class_name TEXT NOT NULL,
                semester TEXT NOT NULL,
                roll_number TEXT NOT NULL,
                student_name TEXT NOT NULL,
                contact_number TEXT,
                face_embeddings_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(class_name, semester, roll_number)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                image_path TEXT,
                embedding_vector TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                admin_id TEXT,
                student_id INTEGER,
                timestamp TEXT NOT NULL,
                device_info TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_students_class ON students(class_name, semester)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_face_data_student ON face_data(student_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_face_logs_student ON face_logs(student_id, timestamp)")


@contextmanager
def student_records_connection():
    initialize_student_records_database()
    conn = sqlite3.connect(str(ANALYTICS_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def embedding_to_json(embedding: np.ndarray) -> str:
    return json.dumps([float(value) for value in np.asarray(embedding).reshape(-1).tolist()])


def save_student_record(record: dict[str, Any]) -> int:
    timestamp = now_text()
    payload = {
        "academic_year": str(record.get("academic_year", "")).strip(),
        "class_name": str(record.get("class_name", "")).strip(),
        "semester": str(record.get("semester", "")).strip(),
        "roll_number": str(record.get("roll_number", "")).strip(),
        "student_name": str(record.get("student_name", "")).strip(),
        "contact_number": str(record.get("contact_number", "")).strip(),
        "face_embeddings_id": str(record.get("face_embeddings_id", "")).strip(),
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with student_records_connection() as conn:
        existing = conn.execute(
            """
            SELECT id FROM students
            WHERE class_name = ? AND semester = ? AND roll_number = ?
            """,
            (payload["class_name"], payload["semester"], payload["roll_number"]),
        ).fetchone()
        if existing:
            payload["id"] = int(existing["id"])
            conn.execute(
                """
                UPDATE students
                SET academic_year = :academic_year,
                    student_name = :student_name,
                    contact_number = :contact_number,
                    face_embeddings_id = :face_embeddings_id,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                payload,
            )
            return int(existing["id"])

        cur = conn.execute(
            """
            INSERT INTO students (
                academic_year, class_name, semester, roll_number, student_name,
                contact_number, face_embeddings_id, created_at, updated_at
            )
            VALUES (
                :academic_year, :class_name, :semester, :roll_number, :student_name,
                :contact_number, :face_embeddings_id, :created_at, :updated_at
            )
            """,
            payload,
        )
        return int(cur.lastrowid)


def get_or_create_student_for_face(class_name: str, semester: str, student_name: str) -> int:
    timestamp = now_text()
    roll_number = str(student_name).strip()
    with student_records_connection() as conn:
        row = conn.execute(
            """
            SELECT id FROM students
            WHERE class_name = ? AND semester = ? AND (student_name = ? OR roll_number = ?)
            ORDER BY id DESC LIMIT 1
            """,
            (class_name, semester, student_name, roll_number),
        ).fetchone()
        if row:
            return int(row["id"])

        cur = conn.execute(
            """
            INSERT INTO students (
                academic_year, class_name, semester, roll_number, student_name,
                contact_number, face_embeddings_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, '', '', ?, ?)
            """,
            (str(datetime.now().year), class_name, semester, roll_number, student_name, timestamp, timestamp),
        )
        return int(cur.lastrowid)


def list_student_records(class_name: str | None = None) -> list[dict[str, Any]]:
    with student_records_connection() as conn:
        params: list[Any] = []
        where = ""
        if class_name:
            where = "WHERE s.class_name = ?"
            params.append(class_name)
        rows = conn.execute(
            f"""
            SELECT
                s.*,
                COUNT(fd.id) AS face_count,
                MIN(fd.image_path) AS preview_path
            FROM students s
            LEFT JOIN face_data fd ON fd.student_id = s.id
            {where}
            GROUP BY s.id
            ORDER BY s.class_name, s.semester, s.roll_number, s.student_name
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def list_face_logs(limit: int = 300) -> list[dict[str, Any]]:
    with student_records_connection() as conn:
        rows = conn.execute(
            """
            SELECT fl.id, fl.action_type, fl.admin_id, fl.student_id, fl.timestamp, fl.device_info,
                   s.roll_number, s.student_name, s.class_name, s.semester
            FROM face_logs fl
            LEFT JOIN students s ON s.id = fl.student_id
            ORDER BY fl.timestamp DESC, fl.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def add_face_data(student_id: int, image_path: str | Path, embedding: np.ndarray, admin_id: str = "admin") -> int:
    timestamp = now_text()
    with student_records_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO face_data (student_id, image_path, embedding_vector, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(student_id), str(image_path), embedding_to_json(embedding), timestamp, timestamp),
        )
        face_id = int(cur.lastrowid)
        embeddings_id = f"student-{int(student_id)}"
        conn.execute(
            "UPDATE students SET face_embeddings_id = ?, updated_at = ? WHERE id = ?",
            (embeddings_id, timestamp, int(student_id)),
        )
        conn.execute(
            """
            INSERT INTO face_logs (action_type, admin_id, student_id, timestamp, device_info)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("upload", admin_id, int(student_id), timestamp, "local-ui"),
        )
    return face_id


def delete_face_data(student_id: int, admin_id: str = "admin") -> list[str]:
    timestamp = now_text()
    with student_records_connection() as conn:
        rows = conn.execute(
            "SELECT image_path FROM face_data WHERE student_id = ?",
            (int(student_id),),
        ).fetchall()
        paths = [str(row["image_path"]) for row in rows if row["image_path"]]
        conn.execute("DELETE FROM face_data WHERE student_id = ?", (int(student_id),))
        conn.execute(
            "UPDATE students SET face_embeddings_id = '', updated_at = ? WHERE id = ?",
            (timestamp, int(student_id)),
        )
        conn.execute(
            """
            INSERT INTO face_logs (action_type, admin_id, student_id, timestamp, device_info)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("delete", admin_id, int(student_id), timestamp, "local-ui"),
        )
    return paths


def log_face_action(action_type: str, student_id: int, admin_id: str = "admin", device_info: str = "local-ui") -> None:
    with student_records_connection() as conn:
        conn.execute(
            """
            INSERT INTO face_logs (action_type, admin_id, student_id, timestamp, device_info)
            VALUES (?, ?, ?, ?, ?)
            """,
            (action_type, admin_id, int(student_id), now_text(), device_info),
        )
