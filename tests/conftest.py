"""Shared pytest fixtures for ecommerce-chat-agent tests."""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import pytest

from ecommerce_chat.agent.llm import AgentResponse
from ecommerce_chat.agent.tools import ToolRegistry
from ecommerce_chat.sessions import ChatMessage


class FakeChatAgent:
    """Deterministic fake LLM agent that exercises ToolRegistry callbacks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_reply(
        self,
        history: list[ChatMessage],
        user_message: str,
        tools: ToolRegistry,
    ) -> AgentResponse:
        self.calls.append(user_message)
        lowered = user_message.lower()
        tool_calls_made: list[dict[str, Any]] = []

        if "order" in lowered:
            output = await tools.execute("get_order_status", {"order_id": "ORD-100"})
            tool_calls_made.append(
                {
                    "tool": "get_order_status",
                    "arguments": {"order_id": "ORD-100"},
                    "output": output,
                }
            )
            return AgentResponse(
                content=f"Order status: {output}",
                tool_calls_made=tool_calls_made,
            )

        if "search" in lowered or "shoes" in lowered:
            output = await tools.execute("search_products", {"query": "shoes"})
            tool_calls_made.append(
                {
                    "tool": "search_products",
                    "arguments": {"query": "shoes"},
                    "output": output,
                }
            )
            return AgentResponse(
                content=f"Found products: {output}",
                tool_calls_made=tool_calls_made,
            )

        return AgentResponse(content=f"AI echo: {user_message}")


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    """Provide an isolated async FakeRedis instance per test."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def fake_agent() -> FakeChatAgent:
    """Provide a deterministic fake LLM agent."""
    return FakeChatAgent()


@pytest.fixture
def single_tenant_callbacks():
    """Return async search_products and get_order_status callbacks."""

    async def search_products(query: str) -> list[dict[str, Any]]:
        return [{"id": "sku-1", "name": f"Running {query}", "price": 79.99}]

    async def get_order_status(order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "shipped"}

    return search_products, get_order_status


@pytest.fixture
def multi_vendor_callbacks():
    """Return async vendor-scoped callbacks tracking all calls."""
    recorded_calls: list[tuple[str, str, str]] = []

    async def vendor_search_products(vendor_id: str, query: str) -> list[dict[str, Any]]:
        recorded_calls.append(("search", vendor_id, query))
        return [{"id": f"{vendor_id}-sku-1", "vendor": vendor_id, "query": query}]

    async def vendor_get_order_status(vendor_id: str, order_id: str) -> dict[str, Any]:
        recorded_calls.append(("order", vendor_id, order_id))
        return {"order_id": order_id, "vendor": vendor_id, "status": "processing"}

    return vendor_search_products, vendor_get_order_status, recorded_calls
