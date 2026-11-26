"""Integration tests for FastAPI router, WebSocket routing, and all three tiers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ecommerce_chat.config import ChatConfig, PlanTier
from ecommerce_chat.exceptions import ConfigurationError, TierViolationError
from ecommerce_chat.router import create_chat_router
from ecommerce_chat.sessions import ChatMode


def test_config_validation_rules(
    fake_redis, single_tenant_callbacks, multi_vendor_callbacks
) -> None:
    search_cb, order_cb = single_tenant_callbacks
    v_search, v_order, _ = multi_vendor_callbacks

    with pytest.raises(ConfigurationError):
        ChatConfig(tier=PlanTier.FREE)

    with pytest.raises(TierViolationError):
        ChatConfig(
            tier=PlanTier.FREE,
            vendor_search_products=v_search,
            vendor_get_order_status=v_order,
        )

    with pytest.raises(ConfigurationError):
        ChatConfig(
            tier=PlanTier.PRO,
            search_products=search_cb,
            get_order_status=order_cb,
        )

    with pytest.raises(ConfigurationError):
        ChatConfig(tier=PlanTier.ENTERPRISE, redis_client=fake_redis)


def test_free_tier_ai_chat_and_handoff_blocked(
    single_tenant_callbacks, fake_agent
) -> None:
    search_cb, order_cb = single_tenant_callbacks
    config = ChatConfig(
        tier=PlanTier.FREE,
        search_products=search_cb,
        get_order_status=order_cb,
        llm_agent=fake_agent,
    )
    app = FastAPI()
    app.include_router(create_chat_router(config), prefix="/chat")

    with TestClient(app) as client:
        with client.websocket_connect("/chat/ws/chat/free-1") as ws:
            ws.send_json({"content": "search for shoes"})
            reply = ws.receive_json()
            assert reply["role"] == "assistant"
            assert "Running shoes" in reply["content"]
            assert reply["tool_calls"][0]["tool"] == "search_products"

            ws.send_json({"content": "where is my order?"})
            order_reply = ws.receive_json()
            assert "ORD-100" in order_reply["content"]
            assert order_reply["tool_calls"][0]["tool"] == "get_order_status"

            ws.send_json({"content": "/human"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "tier_violation"

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/chat/ws/agent/free-1?agent_id=a1"):
                pass

        handoff_resp = client.post("/chat/sessions/free-1/handoff")
        assert handoff_resp.status_code == 403

        session_resp = client.get("/chat/sessions/free-1")
        assert session_resp.status_code == 200
        assert session_resp.json()["mode"] == "ai"


def test_pro_tier_human_handoff_and_return_to_ai(
    fake_redis, single_tenant_callbacks, fake_agent
) -> None:
    search_cb, order_cb = single_tenant_callbacks
    handoff_events: list[tuple[str, str | None, str | None]] = []

    async def on_handoff(
        session_id: str, vendor_id: str | None, reason: str | None
    ) -> None:
        handoff_events.append((session_id, vendor_id, reason))

    config = ChatConfig(
        tier=PlanTier.PRO,
        redis_client=fake_redis,
        search_products=search_cb,
        get_order_status=order_cb,
        llm_agent=fake_agent,
        on_handoff=on_handoff,
    )
    app = FastAPI()
    app.include_router(create_chat_router(config), prefix="/chat")

    with TestClient(app) as client:
        with client.websocket_connect("/chat/ws/chat/pro-1") as customer_ws:
            customer_ws.send_json({"content": "hello"})
            ai_msg = customer_ws.receive_json()
            assert ai_msg["content"] == "AI echo: hello"
            assert len(fake_agent.calls) == 1

            # Customer escalates to human
            customer_ws.send_json({"content": "/human"})
            mode_msg = customer_ws.receive_json()
            assert mode_msg["type"] == "mode_change"
            assert mode_msg["mode"] == ChatMode.HUMAN.value
            assert len(handoff_events) == 1

            # Human agent joins the session
            with client.websocket_connect(
                "/chat/ws/agent/pro-1?agent_id=agent-99"
            ) as agent_ws:
                joined_notice = customer_ws.receive_json()
                assert joined_notice["type"] == "mode_change"
                assert joined_notice["assigned_agent_id"] == "agent-99"

                agent_joined_notice = agent_ws.receive_json()
                assert agent_joined_notice["type"] == "mode_change"
                assert agent_joined_notice["assigned_agent_id"] == "agent-99"

                # Customer message routes to agent room, NOT to LLM
                customer_ws.send_json({"content": "Where is my refund?"})
                routed_to_agent = agent_ws.receive_json()
                assert routed_to_agent["role"] == "customer"
                assert routed_to_agent["content"] == "Where is my refund?"
                assert len(fake_agent.calls) == 1  # LLM was not called

                # Agent replies to customer
                agent_ws.send_json({"content": "I just processed your refund."})
                routed_to_customer = customer_ws.receive_json()
                assert routed_to_customer["role"] == "agent"
                assert routed_to_customer["sender_id"] == "agent-99"
                assert routed_to_customer["content"] == "I just processed your refund."

                # Agent returns conversation to AI mode
                agent_ws.send_json({"content": "/ai"})
                ai_mode_customer = customer_ws.receive_json()
                ai_mode_agent = agent_ws.receive_json()
                assert ai_mode_customer["mode"] == ChatMode.AI.value
                assert ai_mode_agent["mode"] == ChatMode.AI.value

            # Customer message now goes back to LLM
            customer_ws.send_json({"content": "thanks!"})
            ai_msg_2 = customer_ws.receive_json()
            assert ai_msg_2["content"] == "AI echo: thanks!"
            assert len(fake_agent.calls) == 2


def test_enterprise_tier_multi_vendor_isolation(
    fake_redis, multi_vendor_callbacks, fake_agent
) -> None:
    v_search, v_order, recorded = multi_vendor_callbacks
    config = ChatConfig(
        tier=PlanTier.ENTERPRISE,
        redis_client=fake_redis,
        vendor_search_products=v_search,
        vendor_get_order_status=v_order,
        llm_agent=fake_agent,
    )
    app = FastAPI()
    app.include_router(create_chat_router(config), prefix="/chat")

    with TestClient(app) as client:
        # Missing vendor_id must be rejected
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/chat/ws/chat/shared-sess"):
                pass

        # Vendor A and Vendor B use the same session_id without leaking state or tools
        with (
            client.websocket_connect("/chat/ws/chat/shared-sess?vendor_id=vendor-a") as ws_a,
            client.websocket_connect("/chat/ws/chat/shared-sess?vendor_id=vendor-b") as ws_b,
        ):
            ws_a.send_json({"content": "search shoes"})
            reply_a = ws_a.receive_json()
            assert reply_a["vendor_id"] == "vendor-a"
            assert "vendor-a-sku-1" in reply_a["content"]

            ws_b.send_json({"content": "check order"})
            reply_b = ws_b.receive_json()
            assert reply_b["vendor_id"] == "vendor-b"
            assert "vendor-b" in reply_b["content"]

        assert ("search", "vendor-a", "shoes") in recorded
        assert ("order", "vendor-b", "ORD-100") in recorded

        # Cross-vendor REST lookup fails (different Redis key namespace -> 404)
        resp_a = client.get("/chat/sessions/shared-sess?vendor_id=vendor-a")
        assert resp_a.status_code == 200
        assert len(resp_a.json()["history"]) == 2

        resp_missing_vendor = client.get("/chat/sessions/shared-sess")
        assert resp_missing_vendor.status_code == 403

        resp_other_vendor = client.get("/chat/sessions/shared-sess?vendor_id=vendor-c")
        assert resp_other_vendor.status_code == 404
