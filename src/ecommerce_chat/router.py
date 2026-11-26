"""FastAPI router factory for tiered e-commerce AI & human chat orchestration."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from ecommerce_chat.agent.llm import BaseChatAgent, OpenAIChatAgent
from ecommerce_chat.agent.tools import build_tool_registry
from ecommerce_chat.config import ChatConfig, PlanTier
from ecommerce_chat.exceptions import (
    SessionNotFoundError,
    TierViolationError,
    VendorIsolationError,
)
from ecommerce_chat.multi_vendor.isolation import VendorIsolationGuard
from ecommerce_chat.sessions import (
    ChatMessage,
    ChatMode,
    InMemorySessionStore,
    RedisSessionStore,
    SessionState,
    SessionStore,
)
from ecommerce_chat.websocket import ConnectionManager


class HandoffRequest(BaseModel):
    """Request body for escalating a session to human support."""

    agent_id: str | None = None
    reason: str | None = None


def _build_session_store(config: ChatConfig) -> SessionStore:
    if config.tier == PlanTier.FREE:
        return InMemorySessionStore(tier=PlanTier.FREE)
    return RedisSessionStore(
        tier=config.tier,
        redis_url=config.redis_url,
        redis_client=config.redis_client,
        ttl_seconds=config.session_ttl_seconds,
    )


def _extract_message_content(raw_text: str) -> str:
    """Parse JSON `{"content": "..."}` payloads or fall back to plain text."""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    except json.JSONDecodeError:
        pass
    return raw_text


def _is_command(text: str, commands: tuple[str, ...]) -> bool:
    normalized = text.strip().lower()
    return any(normalized == cmd.lower() for cmd in commands)


def create_chat_router(config: ChatConfig) -> APIRouter:
    """Create an embeddable FastAPI APIRouter configured for the specified PlanTier."""
    store = _build_session_store(config)
    manager = ConnectionManager(tier=config.tier)
    agent: BaseChatAgent = config.llm_agent or OpenAIChatAgent(
        api_key=config.openai_api_key,
        model_name=config.model_name,
        system_prompt=config.system_prompt,
        max_tool_iterations=config.max_tool_iterations,
    )

    router = APIRouter(tags=["ecommerce-chat"])
    # Expose internals on the router for diagnostics and testing
    router.session_store = store  # type: ignore[attr-defined]
    router.connection_manager = manager  # type: ignore[attr-defined]
    router.chat_agent = agent  # type: ignore[attr-defined]

    async def _handle_escalation(
        session_id: str,
        vendor_id: str | None,
        agent_id: str | None = None,
        reason: str | None = None,
    ) -> SessionState:
        state = await store.escalate_to_human(
            session_id, agent_id=agent_id, vendor_id=vendor_id
        )
        if config.on_handoff is not None:
            await config.on_handoff(session_id, vendor_id, reason)
        await manager.broadcast_session(
            session_id,
            {
                "type": "mode_change",
                "mode": ChatMode.HUMAN.value,
                "assigned_agent_id": state.assigned_agent_id,
                "session_id": session_id,
                "vendor_id": vendor_id,
            },
            vendor_id=vendor_id,
        )
        return state

    async def _handle_return_to_ai(
        session_id: str,
        vendor_id: str | None,
    ) -> SessionState:
        state = await store.return_to_ai(session_id, vendor_id=vendor_id)
        await manager.broadcast_session(
            session_id,
            {
                "type": "mode_change",
                "mode": ChatMode.AI.value,
                "assigned_agent_id": None,
                "session_id": session_id,
                "vendor_id": vendor_id,
            },
            vendor_id=vendor_id,
        )
        return state

    @router.websocket("/ws/chat/{session_id}")
    async def customer_chat_ws(
        websocket: WebSocket,
        session_id: str,
        vendor_id: str | None = Query(default=None),
    ) -> None:
        try:
            validated_vendor = VendorIsolationGuard.validate_vendor_id(
                vendor_id, config.tier
            )
        except (TierViolationError, VendorIsolationError) as exc:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason=str(exc)
            )
            return

        await manager.connect_customer(
            websocket, session_id, vendor_id=validated_vendor
        )
        try:
            await store.get_or_create(session_id, vendor_id=validated_vendor)
        except VendorIsolationError as exc:
            await manager.send_to_customer(
                session_id,
                {"type": "error", "code": "vendor_isolation", "detail": str(exc)},
                vendor_id=validated_vendor,
            )
            manager.disconnect_customer(
                websocket, session_id, vendor_id=validated_vendor
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason=str(exc)
            )
            return

        try:
            while True:
                raw_text = await websocket.receive_text()
                content = _extract_message_content(raw_text).strip()
                if not content:
                    continue

                if _is_command(content, config.escalation_commands):
                    if not config.supports_human_handoff:
                        await manager.send_to_customer(
                            session_id,
                            {
                                "type": "error",
                                "code": "tier_violation",
                                "detail": "Human handoff requires Pro or Enterprise tier.",
                            },
                            vendor_id=validated_vendor,
                        )
                        continue
                    await _handle_escalation(
                        session_id,
                        validated_vendor,
                        reason="Customer requested human agent",
                    )
                    continue

                if _is_command(content, config.return_to_ai_commands):
                    if not config.supports_human_handoff:
                        await manager.send_to_customer(
                            session_id,
                            {
                                "type": "error",
                                "code": "tier_violation",
                                "detail": "Mode switching requires Pro or Enterprise tier.",
                            },
                            vendor_id=validated_vendor,
                        )
                        continue
                    await _handle_return_to_ai(session_id, validated_vendor)
                    continue

                customer_msg = ChatMessage(role="customer", content=content)
                state = await store.append_message(
                    session_id, customer_msg, vendor_id=validated_vendor
                )

                if state.mode == ChatMode.HUMAN:
                    await manager.send_to_agent_room(
                        session_id,
                        {
                            "type": "message",
                            "role": "customer",
                            "content": content,
                            "session_id": session_id,
                            "vendor_id": validated_vendor,
                        },
                        vendor_id=validated_vendor,
                    )
                    continue

                tools = build_tool_registry(config, vendor_id=validated_vendor)
                history_before = state.history[:-1]
                reply = await agent.generate_reply(
                    history=history_before,
                    user_message=content,
                    tools=tools,
                )
                assistant_msg = ChatMessage(
                    role="assistant",
                    content=reply.content,
                    metadata={"tool_calls": reply.tool_calls_made},
                )
                await store.append_message(
                    session_id, assistant_msg, vendor_id=validated_vendor
                )
                await manager.send_to_customer(
                    session_id,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": reply.content,
                        "tool_calls": reply.tool_calls_made,
                        "session_id": session_id,
                        "vendor_id": validated_vendor,
                    },
                    vendor_id=validated_vendor,
                )
        except WebSocketDisconnect:
            manager.disconnect_customer(
                websocket, session_id, vendor_id=validated_vendor
            )

    @router.websocket("/ws/agent/{session_id}")
    async def agent_chat_ws(
        websocket: WebSocket,
        session_id: str,
        agent_id: str = Query(..., min_length=1),
        vendor_id: str | None = Query(default=None),
    ) -> None:
        if not config.supports_human_handoff:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Human support agent WebSocket requires Pro or Enterprise tier.",
            )
            return

        try:
            validated_vendor = VendorIsolationGuard.validate_vendor_id(
                vendor_id, config.tier
            )
            await store.get(session_id, vendor_id=validated_vendor)
        except (TierViolationError, VendorIsolationError, SessionNotFoundError) as exc:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason=str(exc)
            )
            return

        await manager.connect_agent(
            websocket, session_id, agent_id=agent_id, vendor_id=validated_vendor
        )
        await _handle_escalation(
            session_id,
            validated_vendor,
            agent_id=agent_id,
            reason="Support agent joined session",
        )

        try:
            while True:
                raw_text = await websocket.receive_text()
                content = _extract_message_content(raw_text).strip()
                if not content:
                    continue

                if _is_command(content, config.return_to_ai_commands):
                    await _handle_return_to_ai(session_id, validated_vendor)
                    continue

                agent_msg = ChatMessage(
                    role="agent",
                    content=content,
                    sender_id=agent_id,
                )
                await store.append_message(
                    session_id, agent_msg, vendor_id=validated_vendor
                )
                await manager.send_to_customer(
                    session_id,
                    {
                        "type": "message",
                        "role": "agent",
                        "sender_id": agent_id,
                        "content": content,
                        "session_id": session_id,
                        "vendor_id": validated_vendor,
                    },
                    vendor_id=validated_vendor,
                )
        except WebSocketDisconnect:
            manager.disconnect_agent(
                websocket, session_id, vendor_id=validated_vendor
            )

    @router.get("/sessions/{session_id}", response_model=SessionState)
    async def get_session(
        session_id: str,
        vendor_id: str | None = Query(default=None),
    ) -> SessionState:
        try:
            return await store.get(session_id, vendor_id=vendor_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TierViolationError, VendorIsolationError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/handoff", response_model=SessionState)
    async def trigger_handoff(
        session_id: str,
        body: HandoffRequest | None = None,
        vendor_id: str | None = Query(default=None),
    ) -> SessionState:
        try:
            validated_vendor = VendorIsolationGuard.validate_vendor_id(
                vendor_id, config.tier
            )
            return await _handle_escalation(
                session_id,
                validated_vendor,
                agent_id=body.agent_id if body else None,
                reason=body.reason if body else None,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TierViolationError, VendorIsolationError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/return-to-ai", response_model=SessionState)
    async def return_session_to_ai(
        session_id: str,
        vendor_id: str | None = Query(default=None),
    ) -> SessionState:
        try:
            validated_vendor = VendorIsolationGuard.validate_vendor_id(
                vendor_id, config.tier
            )
            return await _handle_return_to_ai(session_id, validated_vendor)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TierViolationError, VendorIsolationError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    return router
