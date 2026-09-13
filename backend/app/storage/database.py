from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextlib import contextmanager

from app.schemas.domain import AnalysisResult, AnalysisStatus, TraceStep

logger = logging.getLogger(__name__)


def _get_default_db_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "satquery.db"


class DatabaseStorage:
    """SQLite database for persistent analysis history (SIH ₹0 architecture)."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            self._db_path = _get_default_db_path()
        else:
            self._db_path = Path(db_path).resolve()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes database schema and indexes automatically on startup."""
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        query TEXT NOT NULL,
                        intent TEXT,
                        mode TEXT,
                        status TEXT NOT NULL,
                        summary_answer TEXT,
                        result_json TEXT,
                        metadata_json TEXT
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON analysis_sessions(created_at DESC);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_mode ON analysis_sessions(mode);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions(status);"
                )
                conn.commit()
            logger.info("SatQuery SQLite database initialized at %s", self._db_path)
        except Exception as exc:
            logger.error("Failed to initialize SQLite database at %s: %s", self._db_path, exc)

    def save_session(
        self,
        *,
        session_id: str,
        created_at: datetime | str,
        query: str,
        intent: str | None = None,
        mode: str | None = None,
        status: AnalysisStatus | str = AnalysisStatus.COMPLETED,
        summary_answer: str | None = None,
        result: AnalysisResult | None = None,
        trace: list[TraceStep] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persists or updates an analysis session row."""
        try:
            created_str = (
                created_at.isoformat() if isinstance(created_at, datetime) else str(created_at)
            )
            status_str = status.value if isinstance(status, AnalysisStatus) else str(status)

            result_json = result.model_dump_json() if result is not None else None
            meta_dict = metadata.copy() if metadata else {}

            if result is not None:
                if not summary_answer:
                    summary_answer = result.answer
                if not mode:
                    mode = result.mode.value if hasattr(result.mode, "value") else str(result.mode)
                meta_dict.setdefault("confidence", result.confidence)
                meta_dict.setdefault("evidence_count", len(result.evidence))
                if result.building_detection:
                    meta_dict["building_count"] = result.building_detection.count
                if result.building_temporal_change:
                    meta_dict["new_buildings"] = result.building_temporal_change.new_count
                    meta_dict["removed_buildings"] = result.building_temporal_change.removed_count
                    meta_dict["net_buildings"] = (
                        result.building_temporal_change.after_count
                        - result.building_temporal_change.before_count
                    )
                if result.surface_area_change:
                    meta_dict["surface_domain"] = result.surface_area_change.domain
                    meta_dict["surface_diff_m2"] = result.surface_area_change.difference_m2
                    meta_dict["surface_change_pct"] = result.surface_area_change.percentage_change
                if result.cross_modal:
                    meta_dict["optical_summary"] = result.cross_modal.optical_analysis.summary
                    meta_dict["sar_summary"] = result.cross_modal.sar_analysis.summary
                    meta_dict["co_registration"] = result.cross_modal.co_registration_status

            if trace:
                meta_dict["trace_steps_count"] = len(trace)

            metadata_json = json.dumps(meta_dict, default=str)

            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO analysis_sessions (
                        session_id, created_at, query, intent, mode, status, summary_answer, result_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        status = excluded.status,
                        summary_answer = excluded.summary_answer,
                        result_json = excluded.result_json,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        session_id,
                        created_str,
                        query or "",
                        intent or "",
                        mode or "",
                        status_str,
                        summary_answer or "",
                        result_json,
                        metadata_json,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to save session %s to SQLite: %s", session_id, exc)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieves raw session row by session_id."""
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT session_id, created_at, query, intent, mode, status, summary_answer, result_json, metadata_json
                    FROM analysis_sessions
                    WHERE session_id = ?;
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return dict(row)
        except Exception as exc:
            logger.warning("Failed to retrieve session %s from SQLite: %s", session_id, exc)
            return None

    def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        mode: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lists lightweight session history summaries."""
        try:
            query = """
                SELECT session_id, created_at, query, intent, mode, status, summary_answer, metadata_json
                FROM analysis_sessions
            """
            conditions: list[str] = []
            params: list[Any] = []

            if mode:
                conditions.append("mode = ?")
                params.append(mode)
            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
            params.extend([max(1, min(limit, 200)), max(0, offset)])

            with self._connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            items: list[dict[str, Any]] = []
            for r in rows:
                meta = {}
                if r["metadata_json"]:
                    try:
                        meta = json.loads(r["metadata_json"])
                    except Exception:
                        pass
                items.append(
                    {
                        "session_id": r["session_id"],
                        "created_at": r["created_at"],
                        "query": r["query"],
                        "intent": r["intent"],
                        "mode": r["mode"],
                        "status": r["status"],
                        "summary_answer": r["summary_answer"],
                        "confidence": meta.get("confidence"),
                        "metrics_summary": meta,
                    }
                )
            return items
        except Exception as exc:
            logger.warning("Failed to list sessions from SQLite: %s", exc)
            return []

    def count_sessions(
        self,
        *,
        mode: str | None = None,
        status: str | None = None,
    ) -> int:
        """Counts total matching sessions."""
        try:
            query = "SELECT COUNT(*) FROM analysis_sessions"
            conditions: list[str] = []
            params: list[Any] = []

            if mode:
                conditions.append("mode = ?")
                params.append(mode)
            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            with self._connection() as conn:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("Failed to count sessions from SQLite: %s", exc)
            return 0


db_storage = DatabaseStorage()
