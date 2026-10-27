"""Local-only SQLite storage for aggregate session summaries."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd


def database_path() -> Path:
    """Return a per-user data path outside the project directory."""
    override = os.environ.get("EDUSENSE_DATA_DIR")
    if override:
        base = Path(override).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "EduSenseAI" / "sessions.sqlite3"


class SessionStore:
    """Persist only summary rows, never source media or personal identifiers."""

    def __init__(self, path: Path | None = None, keep_latest: int = 50) -> None:
        if keep_latest < 1:
            raise ValueError("keep_latest must be positive.")
        self.path = path or database_path()
        self.keep_latest = keep_latest

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                source_label TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                attendance_json TEXT NOT NULL
            )"""
        )
        connection.commit()
        return connection

    def save(
        self,
        source_label: str,
        summary: dict[str, object],
        metrics: pd.DataFrame,
        attendance: list[dict[str, object]],
        started_at: str | None = None,
    ) -> str:
        session_id = str(uuid4())
        saved_at = started_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        saved_at,
                        source_label,
                        json.dumps(summary, allow_nan=False),
                        metrics.to_json(orient="records", date_format="iso"),
                        json.dumps(attendance, allow_nan=False),
                    ),
                )
                connection.execute(
                    "DELETE FROM sessions WHERE session_id IN "
                    "(SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
                    (self.keep_latest,),
                )
        return session_id

    def list(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id, started_at, source_label, summary_json "
                "FROM sessions ORDER BY started_at DESC LIMIT ?",
                (self.keep_latest,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "source_label": row["source_label"],
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def load(self, session_id: str) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "started_at": row["started_at"],
            "source_label": row["source_label"],
            "summary": json.loads(row["summary_json"]),
            "metrics": pd.read_json(StringIO(row["metrics_json"]), orient="records"),
            "attendance": json.loads(row["attendance_json"]),
        }

    def clear(self) -> int:
        if not self.path.exists():
            return 0
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute("DELETE FROM sessions")
            return max(0, cursor.rowcount)
