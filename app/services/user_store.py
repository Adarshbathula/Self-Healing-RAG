import sqlite3
import threading
from contextlib import contextmanager

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import hash_password, verify_password

logger = get_logger(__name__)

_lock = threading.Lock()


@contextmanager
def _connect():
    settings = get_settings()
    conn = sqlite3.connect(settings.users_db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
            """
        )
        conn.commit()
    logger.info("User database ready")


def create_user(email: str, password: str) -> dict:
    email = email.strip().lower()

    with _lock, _connect() as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise ValueError("A user with this email already exists")

        user_count = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        role = "admin" if user_count == 0 else "user"

        conn.execute(
            "INSERT INTO users (email, hashed_password, role) VALUES (?, ?, ?)",
            (email, hash_password(password), role),
        )
        conn.commit()

    logger.info("Created user %s with role %s", email, role)
    return {"email": email, "role": role}


def authenticate_user(email: str, password: str) -> dict | None:
    email = email.strip().lower()

    with _connect() as conn:
        row = conn.execute(
            "SELECT email, hashed_password, role FROM users WHERE email = ?", (email,)
        ).fetchone()

    if row is None or not verify_password(password, row["hashed_password"]):
        return None

    return {"email": row["email"], "role": row["role"]}
