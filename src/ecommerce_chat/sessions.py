"""Session state models and Redis / in-memory session stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from ecommerce_chat.config import PlanTier
from ecommerce_chat.exceptions import (
    AgentHandoffError,
    ConfigurationError,
    SessionNotFoundError,
    TierViolationError,
)
from ecommerce_chat.multi_vendor.isolation import VendorIsolationGuard


class ChatMode(str, Enum):
    """Active routing mode for a chat session."""

    AI = "ai"
    HUMAN = "human"


class ChatMessage(BaseModel):
    """Single message stored in a session's conversation history."""

    role: Literal["customer", "assistant", "agent", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Complete persisted state for a customer support chat session."""

    session_id: str
    mode: ChatMode = ChatMode.AI
    assigned_agent_id: str | None = None
    vendor_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore(ABC):
    """Abstract base class for session persistence with tier and vendor enforcement."""

    def __init__(self, tier: PlanTier) -> None:
        self.tier = tier

    @abstractmethod
    async def _load_raw(self, key: str) -> str | None:
        """Load raw JSON session payload by scoped key."""

    @abstractmethod
    async def _save_raw(self, key: str, payload: str) -> None:
        """Persist raw JSON session payload by scoped key."""

    @abstractmethod
    async def _delete_raw(self, key: str) -> None:
        """Delete raw session payload by scoped key."""

    def _scoped_key(self, session_id: str, vendor_id: str | None) -> tuple[str, str | None]:
        validated_vendor = VendorIsolationGuard.validate_vendor_id(vendor_id, self.tier)
        key = VendorIsolationGuard.scope_session_key(session_id, validated_vendor, self.tier)
        return key, validated_vendor

    async def get(self, session_id: str, vendor_id: str | None = None) -> SessionState:
        """Retrieve an existing session, enforcing vendor isolation."""
        key, validated_vendor = self._scoped_key(session_id, vendor_id)
        raw = await self._load_raw(key)
        if raw is None:
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        state = SessionState.model_validate_json(raw)
        VendorIsolationGuard.assert_session_vendor(
            state.vendor_id, validated_vendor, self.tier
        )
        return state

    async def get_or_create(
        self, session_id: str, vendor_id: str | None = None
    ) -> SessionState:
        """Fetch an existing session or initialize a new AI-mode session."""
        key, validated_vendor = self._scoped_key(session_id, vendor_id)
        raw = await self._load_raw(key)
        if raw is not None:
            state = SessionState.model_validate_json(raw)
            VendorIsolationGuard.assert_session_vendor(
                state.vendor_id, validated_vendor, self.tier
            )
            return state

        now = datetime.now(timezone.utc)
        state = SessionState(
            session_id=session_id,
            mode=ChatMode.AI,
            vendor_id=validated_vendor,
            created_at=now,
            updated_at=now,
        )
        await self._save_raw(key, state.model_dump_json())
        return state

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage,
        vendor_id: str | None = None,
    ) -> SessionState:
        """Append a message to session history and persist the updated state."""
        key, validated_vendor = self._scoped_key(session_id, vendor_id)
        state = await self.get_or_create(session_id, vendor_id=validated_vendor)
        state.history.append(message)
        state.updated_at = datetime.now(timezone.utc)
        await self._save_raw(key, state.model_dump_json())
        return state

    async def escalate_to_human(
        self,
        session_id: str,
        agent_id: str | None = None,
        vendor_id: str | None = None,
    ) -> SessionState:
        """Transition a session into human support mode (Pro & Enterprise only)."""
        if self.tier == PlanTier.FREE:
            raise TierViolationError(
                "Live human handoff is not available on the Free tier."
            )
        key, validated_vendor = self._scoped_key(session_id, vendor_id)
        state = await self.get(session_id, vendor_id=validated_vendor)
        state.mode = ChatMode.HUMAN
        if agent_id is not None:
            state.assigned_agent_id = agent_id
        state.updated_at = datetime.now(timezone.utc)
        await self._save_raw(key, state.model_dump_json())
        return state

    async def assign_agent(
        self,
        session_id: str,
        agent_id: str,
        vendor_id: str | None = None,
    ) -> SessionState:
        """Assign a human support agent to a session and ensure human mode."""
        if not agent_id or not agent_id.strip():
            raise AgentHandoffError("agent_id must be a non-empty string.")
        return await self.escalate_to_human(
            session_id, agent_id=agent_id.strip(), vendor_id=vendor_id
        )

    async def return_to_ai(
        self,
        session_id: str,
        vendor_id: str | None = None,
    ) -> SessionState:
        """Return a session from human support mode back to AI mode."""
        if self.tier == PlanTier.FREE:
            raise TierViolationError(
                "Mode switching is not available on the Free tier."
            )
        key, validated_vendor = self._scoped_key(session_id, vendor_id)
        state = await self.get(session_id, vendor_id=validated_vendor)
        state.mode = ChatMode.AI
        state.assigned_agent_id = None
        state.updated_at = datetime.now(timezone.utc)
        await self._save_raw(key, state.model_dump_json())
        return state

    async def get_history(
        self, session_id: str, vendor_id: str | None = None
    ) -> list[ChatMessage]:
        """Return the message history for a session."""
        state = await self.get(session_id, vendor_id=vendor_id)
        return list(state.history)

    async def delete(self, session_id: str, vendor_id: str | None = None) -> None:
        """Delete a session from the store."""
        key, _ = self._scoped_key(session_id, vendor_id)
        await self._delete_raw(key)


class InMemorySessionStore(SessionStore):
    """Lightweight in-memory session store intended for Free tier deployments."""

    def __init__(self, tier: PlanTier = PlanTier.FREE) -> None:
        super().__init__(tier=tier)
        self._data: dict[str, str] = {}

    async def _load_raw(self, key: str) -> str | None:
        return self._data.get(key)

    async def _save_raw(self, key: str, payload: str) -> None:
        self._data[key] = payload

    async def _delete_raw(self, key: str) -> None:
        self._data.pop(key, None)


class RedisSessionStore(SessionStore):
    """Redis-backed session store for Pro and Enterprise tiers."""

    def __init__(
        self,
        tier: PlanTier,
        redis_url: str | None = None,
        redis_client: Redis | None = None,
        ttl_seconds: int = 86400,
    ) -> None:
        super().__init__(tier=tier)
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url:
            self._redis = Redis.from_url(redis_url, decode_responses=True)
        else:
            raise ConfigurationError(
                "RedisSessionStore requires either 'redis_url' or 'redis_client'."
            )
        self.ttl_seconds = ttl_seconds

    async def _load_raw(self, key: str) -> str | None:
        value = await self._redis.get(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    async def _save_raw(self, key: str, payload: str) -> None:
        await self._redis.set(key, payload, ex=self.ttl_seconds)

    async def _delete_raw(self, key: str) -> None:
        await self._redis.delete(key)

    async def close(self) -> None:
        """Close the underlying Redis connection pool."""
        await self._redis.aclose()
