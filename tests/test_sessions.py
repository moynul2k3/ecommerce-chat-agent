"""Unit tests for session stores, tier enforcement, and multi-vendor isolation."""

from __future__ import annotations

import pytest

from ecommerce_chat.config import ChatConfig, PlanTier
from ecommerce_chat.exceptions import (
    AgentHandoffError,
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
)


async def test_in_memory_store_free_tier_lifecycle() -> None:
    store = InMemorySessionStore(tier=PlanTier.FREE)

    with pytest.raises(SessionNotFoundError):
        await store.get("missing-session")

    state = await store.get_or_create("sess-1")
    assert state.session_id == "sess-1"
    assert state.mode == ChatMode.AI
    assert state.vendor_id is None

    updated = await store.append_message(
        "sess-1", ChatMessage(role="customer", content="Hello")
    )
    assert len(updated.history) == 1
    assert updated.history[0].content == "Hello"

    with pytest.raises(TierViolationError):
        await store.escalate_to_human("sess-1")

    with pytest.raises(TierViolationError):
        await store.return_to_ai("sess-1")

    with pytest.raises(TierViolationError):
        await store.get_or_create("sess-2", vendor_id="vendor-a")


async def test_redis_store_pro_tier_handoff_and_return(fake_redis) -> None:
    store = RedisSessionStore(
        tier=PlanTier.PRO, redis_client=fake_redis, ttl_seconds=3600
    )

    state = await store.get_or_create("pro-sess-1")
    assert state.mode == ChatMode.AI

    await store.append_message(
        "pro-sess-1", ChatMessage(role="customer", content="Need help")
    )
    escalated = await store.assign_agent("pro-sess-1", agent_id="agent-7")
    assert escalated.mode == ChatMode.HUMAN
    assert escalated.assigned_agent_id == "agent-7"

    with pytest.raises(AgentHandoffError):
        await store.assign_agent("pro-sess-1", agent_id="   ")

    returned = await store.return_to_ai("pro-sess-1")
    assert returned.mode == ChatMode.AI
    assert returned.assigned_agent_id is None

    history = await store.get_history("pro-sess-1")
    assert len(history) == 1
    assert history[0].content == "Need help"


async def test_redis_store_enterprise_vendor_isolation(fake_redis) -> None:
    store = RedisSessionStore(
        tier=PlanTier.ENTERPRISE, redis_client=fake_redis, ttl_seconds=3600
    )

    with pytest.raises(VendorIsolationError):
        await store.get_or_create("shared-id", vendor_id=None)

    with pytest.raises(VendorIsolationError):
        await store.get_or_create("shared-id", vendor_id="bad:vendor")

    await store.append_message(
        "shared-id",
        ChatMessage(role="customer", content="Vendor A message"),
        vendor_id="vendor-a",
    )
    await store.append_message(
        "shared-id",
        ChatMessage(role="customer", content="Vendor B message"),
        vendor_id="vendor-b",
    )

    state_a = await store.get("shared-id", vendor_id="vendor-a")
    state_b = await store.get("shared-id", vendor_id="vendor-b")

    assert state_a.vendor_id == "vendor-a"
    assert state_b.vendor_id == "vendor-b"
    assert [m.content for m in state_a.history] == ["Vendor A message"]
    assert [m.content for m in state_b.history] == ["Vendor B message"]


async def test_vendor_isolation_guard_wraps_callbacks(
    fake_redis, multi_vendor_callbacks
) -> None:
    v_search, v_order, recorded = multi_vendor_callbacks
    config = ChatConfig(
        tier=PlanTier.ENTERPRISE,
        redis_client=fake_redis,
        vendor_search_products=v_search,
        vendor_get_order_status=v_order,
    )

    search_cb, order_cb = VendorIsolationGuard.wrap_vendor_callbacks(
        config, "acme-corp"
    )
    products = await search_cb("jacket")
    order = await order_cb("ORD-99")

    assert products[0]["vendor"] == "acme-corp"
    assert order["vendor"] == "acme-corp"
    assert recorded == [
        ("search", "acme-corp", "jacket"),
        ("order", "acme-corp", "ORD-99"),
    ]
