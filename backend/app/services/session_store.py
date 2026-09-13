from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import logging

from app.schemas.domain import AnalysisResult, AnalysisStatus, TraceStep, TraceStatus
from app.schemas.region_interpretation import BiTemporalRegionInterpretationResult
from app.storage.database import db_storage

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    turn_id: str
    turn_index: int
    user_message: str
    assistant_answer: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    route: str | None = None
    provider: str | None = None
    scope: str | None = None


@dataclass
class RegionConversation:
    conversation_id: str
    session_id: str
    region_id: str
    turns: list[ConversationTurn] = field(default_factory=list)


@dataclass
class QuerySession:
    session_id: str
    status: AnalysisStatus
    trace: list[TraceStep] = field(default_factory=list)
    result: AnalysisResult | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    query: str = ""
    intent: str | None = None
    mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """Session store with in-memory caching and SQLite persistence."""

    def __init__(self) -> None:
        self._sessions: dict[str, QuerySession] = {}
        self._conversations: dict[str, RegionConversation] = {}
        self._interpretations: dict[str, BiTemporalRegionInterpretationResult] = {}

    @staticmethod
    def _conversation_key(session_id: str, region_id: str) -> str:
        return f"{session_id}:{region_id}"

    def get_conversation(self, session_id: str, region_id: str) -> RegionConversation | None:
        return self._conversations.get(self._conversation_key(session_id, region_id))

    def get_or_create_conversation(self, session_id: str, region_id: str) -> RegionConversation:
        key = self._conversation_key(session_id, region_id)
        existing = self._conversations.get(key)
        if existing is not None:
            return existing
        conversation = RegionConversation(
            conversation_id=str(uuid.uuid4()),
            session_id=session_id,
            region_id=region_id,
        )
        self._conversations[key] = conversation
        return conversation

    def append_turn(self, session_id: str, region_id: str, turn: ConversationTurn) -> RegionConversation:
        conversation = self.get_or_create_conversation(session_id, region_id)
        conversation.turns.append(turn)
        return conversation

    def store_interpretation(
        self,
        session_id: str,
        region_id: str,
        interpretation: BiTemporalRegionInterpretationResult,
    ) -> None:
        self._interpretations[self._conversation_key(session_id, region_id)] = interpretation

    def get_latest_interpretation(
        self,
        session_id: str,
        region_id: str,
    ) -> BiTemporalRegionInterpretationResult | None:
        return self._interpretations.get(self._conversation_key(session_id, region_id))

    def create(self, *args: Any, query: str = "", mode: str = "", intent: str | None = None, **kwargs: Any) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = QuerySession(
            session_id=session_id,
            status=AnalysisStatus.PENDING,
            query=query,
            mode=mode,
            intent=intent,
        )
        return session_id

    def get(self, session_id: str) -> QuerySession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            return session

        # Attempt to reload from SQLite persistence
        try:
            row = db_storage.get_session(session_id)
            if row:
                status_enum = AnalysisStatus.COMPLETED
                try:
                    status_enum = AnalysisStatus(row["status"])
                except Exception:
                    pass

                result_obj = None
                trace_list: list[TraceStep] = []
                if row["result_json"]:
                    try:
                        result_obj = AnalysisResult.model_validate_json(row["result_json"])
                        trace_list = result_obj.trace
                    except Exception as exc:
                        logger.warning("Failed to deserialize result_json for %s: %s", session_id, exc)

                created_dt = datetime.now(UTC)
                if row["created_at"]:
                    try:
                        created_dt = datetime.fromisoformat(row["created_at"])
                    except Exception:
                        pass

                reconstructed = QuerySession(
                    session_id=row["session_id"],
                    status=status_enum,
                    trace=trace_list,
                    result=result_obj,
                    created_at=created_dt,
                    query=row["query"] or "",
                    intent=row["intent"] or None,
                    mode=row["mode"] or None,
                )
                self._sessions[session_id] = reconstructed
                return reconstructed
        except Exception as exc:
            logger.warning("Error reloading session %s from SQLite: %s", session_id, exc)

        return None

    def update_trace(self, session_id: str, trace: list[TraceStep]) -> None:
        session = self._require(session_id)
        session.trace = trace

    def complete(
        self,
        session_id: str,
        result: AnalysisResult,
        *,
        query: str | None = None,
        intent: str | None = None,
        mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        session = self._require(session_id)
        session.status = result.status
        session.result = result
        session.trace = result.trace
        if query:
            session.query = query
        if intent:
            session.intent = intent
        if mode:
            session.mode = mode

        # Persist to SQLite
        try:
            db_storage.save_session(
                session_id=session_id,
                created_at=session.created_at,
                query=session.query or query or "",
                intent=session.intent or intent or None,
                mode=session.mode
                or mode
                or (result.mode.value if hasattr(result.mode, "value") else str(result.mode)),
                status=result.status,
                summary_answer=result.answer,
                result=result,
                trace=result.trace,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to persist completed session %s: %s", session_id, exc)

    def list_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        mode: str | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Returns (items, total_count) from persistent storage."""
        items = db_storage.list_sessions(limit=limit, offset=offset, mode=mode, status=status)
        total = db_storage.count_sessions(mode=mode, status=status)
        return items, total

    def _require(self, session_id: str) -> QuerySession:
        session = self.get(session_id)
        if not session:
            raise KeyError(session_id)
        return session


session_store = SessionStore()
