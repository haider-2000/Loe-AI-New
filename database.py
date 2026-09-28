from __future__ import annotations

import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import aiosqlite

try:  # optional: only needed when the data lives in a remote libSQL database
    from libsql_client import create_client as _create_client
except ImportError:  # pragma: no cover - a laptop-only install never needs it
    _create_client = None  # type: ignore[assignment]


SCHEMA = """
CREATE TABLE IF NOT EXISTS contributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK(type IN ('text', 'image')),
    file_path TEXT,
    text_content TEXT,
    transcription TEXT,
    model_answer TEXT,
    language TEXT,
    dialect TEXT,
    privacy_flag INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'review',
    created_at TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_contributions_status ON contributions(status);
CREATE INDEX IF NOT EXISTS idx_contributions_created_at ON contributions(created_at);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    lesson_date TEXT NOT NULL,
    lesson_time TEXT NOT NULL,
    capacity INTEGER NOT NULL DEFAULT 50 CHECK(capacity BETWEEN 1 AND 50),
    created_at TEXT NOT NULL,
    UNIQUE(title, lesson_date, lesson_time)
);
CREATE TABLE IF NOT EXISTS lesson_students (
    lesson_id INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    student_code TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    PRIMARY KEY(lesson_id, student_code)
);
CREATE INDEX IF NOT EXISTS idx_lesson_students_lesson ON lesson_students(lesson_id);
CREATE TABLE IF NOT EXISTS images (
    name TEXT PRIMARY KEY,
    content BLOB NOT NULL,
    stored_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    questions TEXT NOT NULL,
    total INTEGER NOT NULL,
    score INTEGER NOT NULL DEFAULT 0,
    answered INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_student
    ON quiz_sessions(chat_id, user_id, created_at);
"""

REMOTE_PREFIXES = ("libsql://", "https://", "wss://")
_TABLES = ("contributions", "lessons", "lesson_students", "images", "quiz_sessions")


def database_url() -> str:
    """Where the data lives.

    A plain path means a local SQLite file on this machine. A libsql:// URL
    means a remote database, which is what keeps the data alive on Render: the
    container filesystem there is wiped on every restart, a remote database is
    not. DB_URL wins over DATABASE_PATH so the old variable keeps working.
    """
    return (os.getenv("DB_URL") or os.getenv("DATABASE_PATH") or "data/edu_bot.db").strip()


def is_remote(url: str | None = None) -> bool:
    return (database_url() if url is None else url).startswith(REMOTE_PREFIXES)


def _statements(script: str) -> list[str]:
    return [chunk.strip() for chunk in script.split(";") if chunk.strip()]


def _hash_payload(*parts: str | None) -> str:
    raw = "\x1f".join(part or "" for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _RemoteCursor:
    """The slice of the aiosqlite cursor API this module relies on.

    Rows arrive as dicts, which is what aiosqlite.Row also behaves like, so the
    callers above cannot tell the two backends apart.
    """

    def __init__(self, result: Any) -> None:
        columns: Sequence[str] = getattr(result, "columns", None) or ()
        rows: list[dict[str, Any]] = []
        for raw in getattr(result, "rows", None) or ():
            # libSQL hands back Row objects that can describe themselves; a plain
            # sequence of values is matched against the column names instead.
            as_dict = getattr(raw, "asdict", None)
            rows.append(as_dict() if callable(as_dict) else dict(zip(columns, raw)))
        self._rows = rows
        self._next = 0
        self.rowcount = int(getattr(result, "rows_affected", 0) or 0)
        raw_id = getattr(result, "last_insert_rowid", 0)
        last_id = raw_id() if callable(raw_id) else raw_id
        self.lastrowid = int(last_id) if last_id else None

    async def fetchone(self) -> dict[str, Any] | None:
        if self._next >= len(self._rows):
            return None
        row = self._rows[self._next]
        self._next += 1
        return row

    async def fetchall(self) -> list[dict[str, Any]]:
        rows = self._rows[self._next:]
        self._next = len(self._rows)
        return rows

    async def __aenter__(self) -> "_RemoteCursor":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _RemoteConn:
    """Minimal async connection facade over the libSQL client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> _RemoteCursor:
        args = list(params) if params else None
        return _RemoteCursor(await self._client.execute(sql, args))

    async def commit(self) -> None:
        # libSQL autocommits every statement, so there is nothing to flush.
        return None

    async def close(self) -> None:
        closer = self._client.close()
        if closer is not None and hasattr(closer, "__await__"):
            await closer


def remote_transport_url(url: str) -> str:
    """Point the libSQL client at the HTTP API.

    Turso's dashboard hands out a libsql:// URL, and libsql-client turns that
    into a wss:// WebSocket handshake, which the hosted endpoint refuses with a
    400. Its HTTP API speaks the same protocol, so switch the scheme.
    """
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://"):]
    return url


@asynccontextmanager
async def _connect(url: str) -> AsyncIterator[Any]:
    if is_remote(url):
        if _create_client is None:
            raise RuntimeError(
                "DB_URL points at a remote database but libsql-client is not "
                "installed. Run: pip install libsql-client"
            )
        client = _create_client(remote_transport_url(url),
                                auth_token=os.getenv("DB_AUTH_TOKEN") or None)
        connection = _RemoteConn(client)
        try:
            yield connection
        finally:
            await connection.close()
        return
    # SQLite ignores foreign keys unless the pragma is set per connection, and
    # aiosqlite does not do it for us, so ON DELETE CASCADE would be inert.
    db = await aiosqlite.connect(url, timeout=30.0)
    db.row_factory = aiosqlite.Row
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        # Many students can be answered at once, so a writer must wait its turn
        # instead of raising "database is locked" straight away.
        await db.execute("PRAGMA busy_timeout = 30000")
        yield db
    finally:
        await db.close()


async def _fetch_all(db: Any, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    cursor = await db.execute(sql, tuple(params))
    async with cursor:
        return [dict(row) for row in await cursor.fetchall()]


async def init_db(url: str) -> None:
    if not is_remote(url):
        Path(url).parent.mkdir(parents=True, exist_ok=True)
    async with _connect(url) as db:
        if not is_remote(url):
            # WAL lets readers work while a write is in flight, which is what keeps
            # a busy group from serialising every request behind one write.
            await db.execute("PRAGMA journal_mode = WAL")
        for statement in _statements(SCHEMA):
            await db.execute(statement)
        await db.commit()


async def add_contribution(
    url: str,
    *,
    contribution_type: str,
    file_path: str | None,
    text_content: str | None,
    transcription: str | None,
    model_answer: str | None,
    language: str,
    dialect: str,
    privacy_flag: bool,
    status: str = "review",
    dedupe_key: str | None = None,
) -> bool:
    if contribution_type not in {"text", "image"}:
        raise ValueError("Only text and image contributions are allowed")
    # file_path is a per-upload random name, so hashing it would make every
    # upload unique; dedupe_key carries a stable identity for that case.
    identity = dedupe_key if dedupe_key is not None else file_path
    content_hash = _hash_payload(contribution_type, identity, text_content, transcription, model_answer)
    created_at = datetime.now(timezone.utc).isoformat()
    async with _connect(url) as db:
        cursor = await db.execute(
            """INSERT OR IGNORE INTO contributions
            (type, file_path, text_content, transcription, model_answer, language,
             dialect, privacy_flag, status, created_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (contribution_type, file_path, text_content, transcription, model_answer,
             language, dialect, int(privacy_flag), status, created_at, content_hash),
        )
        await db.commit()
        return cursor.rowcount == 1


async def count_by_status(url: str) -> dict[str, int]:
    async with _connect(url) as db:
        rows = await _fetch_all(db, "SELECT status, COUNT(*) AS n FROM contributions GROUP BY status")
        return {str(row["status"]): int(row["n"]) for row in rows}


async def count_contributions(url: str) -> int:
    async with _connect(url) as db:
        rows = await _fetch_all(db, "SELECT COUNT(*) AS n FROM contributions")
        return int(rows[0]["n"]) if rows else 0


async def pending_rows(url: str, limit: int = 50) -> list[dict[str, Any]]:
    async with _connect(url) as db:
        return await _fetch_all(
            db,
            """SELECT id, type, text_content, transcription, model_answer,
               language, dialect, privacy_flag, status, created_at FROM contributions
               WHERE status != 'approved' ORDER BY id DESC LIMIT ?""", (limit,))


async def approved_rows(url: str) -> list[dict[str, Any]]:
    async with _connect(url) as db:
        return await _fetch_all(
            db,
            """SELECT type, file_path, text_content, transcription, model_answer,
               language, dialect FROM contributions
               WHERE status = 'approved' AND privacy_flag = 0 ORDER BY id""")


async def set_privacy_flag(url: str, file_path: str) -> int:
    """Flag a stored contribution by file path (used after a visual PII check)."""
    async with _connect(url) as db:
        cursor = await db.execute(
            "UPDATE contributions SET privacy_flag = 1 WHERE file_path = ?", (file_path,)
        )
        await db.commit()
        return cursor.rowcount


async def store_image(url: str, name: str, data: bytes) -> None:
    """Keep the image bytes next to the rows that point at them.

    On Render the disk is wiped on every restart, so a row whose image is gone
    is a broken record. Writing the bytes into the database makes the whole
    dataset survive a restart.
    """
    async with _connect(url) as db:
        await db.execute(
            "INSERT OR REPLACE INTO images (name, content, stored_at) VALUES (?, ?, ?)",
            (name, data, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()


async def load_image(url: str, name: str) -> bytes | None:
    async with _connect(url) as db:
        rows = await _fetch_all(db, "SELECT content FROM images WHERE name = ?", (name,))
    if not rows:
        return None
    content = rows[0]["content"]
    return bytes(content) if content is not None else None


async def image_names(url: str) -> list[str]:
    async with _connect(url) as db:
        rows = await _fetch_all(db, "SELECT name FROM images ORDER BY name")
    return [str(row["name"]) for row in rows]


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif")


async def backfill_images_from_disk(url: str, image_dir: str = "data/raw/images") -> list[str]:
    """Pull image files into the database, for records that predate the table.

    Contributions made before images existed only point at a file on disk, so
    copying the tables on their own would leave those rows broken as soon as the
    laptop stops being the thing that serves them.
    """
    directory = Path(image_dir)
    if not directory.is_dir():
        return []
    already = set(await image_names(url))
    added: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name in already:
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        await store_image(url, path.name, path.read_bytes())
        added.append(path.name)
    return added


async def snapshot_database(url: str, dest: str) -> None:
    """Write a consistent copy of the live database to dest.

    For a local file this uses SQLite's online backup API, which is safe while
    the bot is serving requests. A remote database has no file to copy, so the
    tables are replayed into a fresh local SQLite file instead.
    """
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if not is_remote(url):
        source = await aiosqlite.connect(url)
        try:
            target = await aiosqlite.connect(dest)
            try:
                await source.backup(target)
                await target.commit()
            finally:
                await target.close()
        finally:
            await source.close()
        return
    target = await aiosqlite.connect(dest)
    try:
        for statement in _statements(SCHEMA):
            await target.execute(statement)
        async with _connect(url) as db:
            for table in _TABLES:
                rows = await _fetch_all(db, f"SELECT * FROM {table}")
                if not rows:
                    continue
                columns = list(rows[0].keys())
                await target.executemany(
                    f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
                    f"VALUES ({','.join('?' * len(columns))})",
                    [tuple(row[column] for column in columns) for row in rows])
        await target.commit()
    finally:
        await target.close()


async def migrate_database(source_url: str, dest_url: str) -> dict[str, int]:
    """Copy every table from one database into another, keeping the row ids.

    This is how the data that is already on the laptop reaches a fresh remote
    database, so a first deploy does not come up with an empty dataset.
    """
    if source_url == dest_url:
        raise ValueError("the source and the destination are the same database")
    moved: dict[str, int] = {}
    async with _connect(source_url) as source, _connect(dest_url) as destination:
        for table in _TABLES:
            rows = await _fetch_all(source, f"SELECT * FROM {table}")
            if not rows:
                moved[table] = 0
                continue
            columns = list(rows[0].keys())
            statement = (f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
                         f"VALUES ({','.join('?' * len(columns))})")
            for row in rows:
                await destination.execute(statement, tuple(row[column] for column in columns))
            await destination.commit()
            moved[table] = len(rows)
    return moved


async def export_jsonl(url: str, output_path: str) -> tuple[str, int]:
    rows = await approved_rows(url)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            item = {
                "type": row["type"],
                "text": row["text_content"] or row["transcription"] or "",
                "answer": row["model_answer"] or "",
                "language": row["language"] or "ar",
                "dialect": row["dialect"] or "iraqi",
            }
            if row["type"] == "image":
                item["image"] = row["file_path"] or ""
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path, len(rows)


async def create_lesson(url: str, title: str, lesson_date: str, lesson_time: str, capacity: int = 50) -> int:
    if not 1 <= capacity <= 50:
        raise ValueError("capacity must be between 1 and 50")
    async with _connect(url) as db:
        cursor = await db.execute(
            "INSERT INTO lessons (title, lesson_date, lesson_time, capacity, created_at) VALUES (?, ?, ?, ?, ?)",
            (title.strip(), lesson_date.strip(), lesson_time.strip(), capacity,
             datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
        return int(cursor.lastrowid or 0)


async def list_lessons(url: str) -> list[dict[str, Any]]:
    async with _connect(url) as db:
        return await _fetch_all(
            db,
            """SELECT l.id, l.title, l.lesson_date, l.lesson_time, l.capacity,
                      COUNT(ls.student_code) AS enrolled
               FROM lessons l LEFT JOIN lesson_students ls ON ls.lesson_id = l.id
               GROUP BY l.id ORDER BY l.lesson_date, l.lesson_time, l.id""")


async def enroll_student(url: str, lesson_id: int, student_code: str) -> tuple[bool, str]:
    student_code = student_code.strip()
    if not student_code or len(student_code) > 40:
        return False, "رمز الطالب غير صالح."
    async with _connect(url) as db:
        lessons = await _fetch_all(db, "SELECT capacity FROM lessons WHERE id = ?", (lesson_id,))
        if not lessons:
            return False, "الحصة غير موجودة."
        already = await _fetch_all(
            db, "SELECT 1 AS hit FROM lesson_students WHERE lesson_id = ? AND student_code = ?",
            (lesson_id, student_code))
        if already:
            return False, "الطالب مسجل بهذه الحصة مسبقًا."
        # The seat check lives inside the INSERT, so the limit cannot be
        # overshot when two students register at the same moment.
        cursor = await db.execute(
            """INSERT INTO lesson_students (lesson_id, student_code, registered_at)
               SELECT ?, ?, ?
               WHERE (SELECT COUNT(*) FROM lesson_students WHERE lesson_id = ?)
                     < (SELECT capacity FROM lessons WHERE id = ?)""",
            (lesson_id, student_code, datetime.now(timezone.utc).isoformat(), lesson_id, lesson_id),
        )
        await db.commit()
        if cursor.rowcount != 1:
            return False, "الحصة مكتملة (الحد الأعلى ٥٠ طالب)."
        counts = await _fetch_all(
            db, "SELECT COUNT(*) AS n FROM lesson_students WHERE lesson_id = ?", (lesson_id,))
        enrolled = int(counts[0]["n"]) if counts else 1
        return True, f"تم التسجيل. المقاعد المتبقية: {int(lessons[0]['capacity']) - enrolled}"


async def lesson_roster(url: str, lesson_id: int) -> dict[str, Any] | None:
    async with _connect(url) as db:
        rows = await _fetch_all(
            db,
            """SELECT l.id, l.title, l.lesson_date, l.lesson_time, l.capacity,
                      ls.student_code
               FROM lessons l LEFT JOIN lesson_students ls ON ls.lesson_id = l.id
               WHERE l.id = ? ORDER BY ls.student_code""", (lesson_id,))
    if not rows:
        return None
    first = rows[0]
    return {"id": first["id"], "title": first["title"], "lesson_date": first["lesson_date"],
            "lesson_time": first["lesson_time"], "capacity": first["capacity"],
            "students": [row["student_code"] for row in rows if row["student_code"] is not None]}


async def create_quiz_session(url: str, *, chat_id: int, user_id: int, topic: str,
                              questions: str, total: int) -> int:
    """Start a quiz and return its id.

    The questions live in the database rather than in memory because the answer
    buttons come back later as callbacks: a service restart in between, which
    Render does often, would otherwise leave the student with buttons that no
    longer mean anything.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    async with _connect(url) as db:
        cursor = await db.execute(
            """INSERT INTO quiz_sessions
            (chat_id, user_id, topic, questions, total, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (chat_id, user_id, topic, questions, total, created_at))
        await db.commit()
        return int(cursor.lastrowid)


async def get_quiz_session(url: str, session_id: int) -> dict[str, Any] | None:
    async with _connect(url) as db:
        rows = await _fetch_all(
            db,
            """SELECT id, chat_id, user_id, topic, questions, total, score,
                      answered, created_at, finished_at
               FROM quiz_sessions WHERE id = ?""", (session_id,))
    return rows[0] if rows else None


async def record_quiz_answer(url: str, session_id: int, correct: bool) -> dict[str, Any] | None:
    """Bank one answer and hand back the updated score.

    The update adds to whatever is already banked, so a replayed button tap
    counts once: a student who presses the same answer twice, or a client that
    retries, cannot inflate a score. Reaching the total closes the session.
    """
    async with _connect(url) as db:
        cursor = await db.execute(
            """UPDATE quiz_sessions
               SET score = score + ?, answered = answered + 1
               WHERE id = ? AND answered < total""",
            (int(correct), session_id))
        await db.commit()
        if cursor.rowcount == 0:
            # Either the session is gone or every answer is already in, so the
            # score must not move.
            rows = await _fetch_all(
                db, "SELECT id, score, answered, total, finished_at FROM quiz_sessions WHERE id = ?",
                (session_id,))
            return rows[0] if rows else None
        rows = await _fetch_all(
            db, "SELECT id, score, answered, total, finished_at FROM quiz_sessions WHERE id = ?",
            (session_id,))
        if not rows:
            return None
        row = rows[0]
        if row["answered"] >= row["total"]:
            await db.execute("UPDATE quiz_sessions SET finished_at = ? WHERE id = ?",
                             (datetime.now(timezone.utc).isoformat(), session_id))
            await db.commit()
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
        return row


async def quiz_history(url: str, chat_id: int, user_id: int,
                       limit: int = 10) -> list[dict[str, Any]]:
    async with _connect(url) as db:
        return await _fetch_all(
            db,
            """SELECT topic, score, total, answered, created_at, finished_at
               FROM quiz_sessions
               WHERE chat_id = ? AND user_id = ?
               ORDER BY id DESC LIMIT ?""", (chat_id, user_id, limit))


async def quiz_overall(url: str, chat_id: int, user_id: int) -> dict[str, int]:
    """Totals for one student, which is what /progress reports."""
    async with _connect(url) as db:
        rows = await _fetch_all(
            db,
            """SELECT COUNT(*) AS quizzes, COALESCE(SUM(total), 0) AS questions,
                      COALESCE(SUM(score), 0) AS correct,
                      COALESCE(SUM(answered), 0) AS answered
               FROM quiz_sessions WHERE chat_id = ? AND user_id = ?""",
            (chat_id, user_id))
    row = rows[0] if rows else {}
    return {"quizzes": int(row.get("quizzes") or 0),
            "questions": int(row.get("questions") or 0),
            "correct": int(row.get("correct") or 0),
            "answered": int(row.get("answered") or 0)}
