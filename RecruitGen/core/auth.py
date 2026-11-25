"""User authentication, admin account, activity log, login sessions and screening history (SQLite)."""

import os
import re
import sqlite3
import hashlib
import secrets
import uuid
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "recruitgen.db")

# Built-in administrator (override with environment variables in production).
ADMIN_USERNAME = os.environ.get("RECRUITGEN_ADMIN_USER", "admin_recruitgen")
ADMIN_PASSWORD = os.environ.get("RECRUITGEN_ADMIN_PASSWORD", "RecruitGen@2026")
ADMIN_EMAIL = os.environ.get("RECRUITGEN_ADMIN_EMAIL", "admin@recruitgen.app")
ADMIN_NAME = "RecruitGen Admin"


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _connect():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                candidate TEXT,
                file_name TEXT,
                predicted_role TEXT,
                confidence REAL,
                top_fit_role TEXT,
                fit_score REAL,
                ai_score REAL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                full_name TEXT,
                event TEXT NOT NULL,
                details TEXT,
                session_id TEXT,
                client TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                username TEXT,
                full_name TEXT,
                client TEXT,
                login_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                logout_at TEXT
            )"""
        )
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        if "role" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if not conn.execute("SELECT 1 FROM users WHERE username = ?", (ADMIN_USERNAME,)).fetchone():
            salt = secrets.token_hex(16)
            conn.execute(
                "INSERT INTO users (full_name, username, email, password_hash, salt, created_at, role) "
                "VALUES (?,?,?,?,?,?,'admin')",
                (ADMIN_NAME, ADMIN_USERNAME, ADMIN_EMAIL, _hash(ADMIN_PASSWORD, salt), salt, _now()),
            )
        conn.execute("UPDATE users SET role = 'admin' WHERE username = ?", (ADMIN_USERNAME,))


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def valid_email(email):
    return re.match(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$", email or "") is not None


def create_user(full_name, username, email, password):
    full_name, username, email = full_name.strip(), username.strip().lower(), email.strip().lower()
    if not all([full_name, username, email, password]):
        return False, "Please fill in all fields."
    if not re.match(r"^[a-z0-9_]{3,20}$", username):
        return False, "Username must be 3-20 characters (letters, numbers, underscore)."
    if not valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (full_name, username, email, password_hash, salt, created_at) VALUES (?,?,?,?,?,?)",
                (full_name, username, email, _hash(password, salt), salt, _now()),
            )
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def authenticate(identifier, password):
    identifier = (identifier or "").strip().lower()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?", (identifier, identifier)).fetchone()
    if row and secrets.compare_digest(row["password_hash"], _hash(password, row["salt"])):
        return {"id": row["id"], "full_name": row["full_name"], "username": row["username"], "email": row["email"],
                "role": row["role"] or "user"}
    return None


def is_admin(user):
    return bool(user) and user.get("role") == "admin"


def save_screening(user_id, candidate, file_name, predicted_role, confidence, top_fit_role, fit_score, ai_score=None):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO screenings (user_id, candidate, file_name, predicted_role, confidence,
               top_fit_role, fit_score, ai_score, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (user_id, candidate, file_name, predicted_role, confidence, top_fit_role, fit_score, ai_score,
             _now()),
        )


def get_screenings(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM screenings WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_screenings(user_id):
    with _connect() as conn:
        conn.execute("DELETE FROM screenings WHERE user_id = ?", (user_id,))


def get_all_screenings():
    with _connect() as conn:
        rows = conn.execute(
            """SELECT s.*, u.username, u.full_name AS recruiter FROM screenings s
               LEFT JOIN users u ON u.id = s.user_id ORDER BY s.id DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def clear_all_screenings():
    with _connect() as conn:
        conn.execute("DELETE FROM screenings")


# ---------------- activity log & login sessions ----------------

def log_activity(user, event, details="", session_id=None, client="", username=None):
    """Record one event. `user` may be None (e.g. failed login) — pass `username` instead."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO activity (user_id, username, full_name, event, details, session_id, client, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            ((user or {}).get("id"), (user or {}).get("username", username), (user or {}).get("full_name"),
             event, details, session_id, client, _now()),
        )


def start_session(user, client=""):
    session_id = uuid.uuid4().hex
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO sessions (id, user_id, username, full_name, client, login_at, last_seen)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, user["id"], user["username"], user["full_name"], client, now, now),
        )
    return session_id


def touch_session(session_id):
    with _connect() as conn:
        conn.execute("UPDATE sessions SET last_seen = ? WHERE id = ? AND logout_at IS NULL", (_now(), session_id))


def end_session(session_id):
    now = _now()
    with _connect() as conn:
        conn.execute("UPDATE sessions SET logout_at = ?, last_seen = ? WHERE id = ? AND logout_at IS NULL",
                     (now, now, session_id))


def get_activity():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM activity ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_sessions():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY login_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_users():
    with _connect() as conn:
        rows = conn.execute(
            """SELECT u.id, u.full_name, u.username, u.email, u.role, u.created_at,
                      (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id) AS logins,
                      (SELECT MAX(login_at) FROM sessions s WHERE s.user_id = u.id) AS last_login,
                      (SELECT MAX(last_seen) FROM sessions s WHERE s.user_id = u.id) AS last_seen,
                      (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND s.logout_at IS NULL) AS open_sessions,
                      (SELECT COUNT(*) FROM screenings c WHERE c.user_id = u.id) AS screenings,
                      (SELECT COUNT(*) FROM activity a WHERE a.user_id = u.id) AS events
               FROM users u ORDER BY u.id"""
        ).fetchall()
    return [dict(r) for r in rows]


def clear_activity():
    with _connect() as conn:
        conn.execute("DELETE FROM activity")
