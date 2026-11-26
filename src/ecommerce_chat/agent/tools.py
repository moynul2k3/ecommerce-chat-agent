"""Typed tool definitions with async callback delegation and vendor context injection."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ecommerce_chat.config import (
    ChatConfig,
    GetOrderStatusCallback,
    PlanTier,
    SearchProductsCallback,
)
from ecommerce_chat.exceptions import ToolExecutionError
from ecommerce_chat.multi_vendor.isolation import VendorIsolationGuard


class SearchProductsInput(BaseModel):
    """Input schema for the `search_products` LLM tool."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search keywords, product name, category, or SKU.",
    )


class GetOrderStatusInput(BaseModel):
    """Input schema for the `get_order_status` LLM tool."""

    order_id: str = Field(
        ...,
        min_length=1,
        description="The customer's order identifier (e.g. ORD-12345).",
    )


def _serialize_tool_output(result: Any) -> str:
    """Convert Pydantic models, lists, or dicts into a JSON string for LLM consumption."""
    if result is None:
        return json.dumps({"found": False, "result": None})
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        normalized = [
            item.model_dump() if isinstance(item, BaseModel) else item for item in result
        ]
        return json.dumps(normalized, default=str)
    return json.dumps(result, default=str)


class ToolRegistry:
    """Manages typed e-commerce tools bound to a specific tenant/session context."""

    def __init__(
        self,
        search_products: SearchProductsCallback,
        get_order_status: GetOrderStatusCallback,
        vendor_id: str | None = None,
    ) -> None:
        self._search_products = search_products
        self._get_order_status = get_order_status
        self.vendor_id = vendor_id

    @staticmethod
    def openai_tool_schemas() -> list[dict[str, Any]]:
        """Return OpenAI Chat Completions function-calling tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": (
                        "Search the store's product catalog by keywords, product name, "
                        "category, or SKU."
                    ),
                    "parameters": SearchProductsInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "description": (
                        "Look up the current fulfillment and shipping status of an order "
                        "by its order ID."
                    ),
                    "parameters": GetOrderStatusInput.model_json_schema(),
                },
            },
        ]

    def langchain_tools(self) -> list[Any]:
        """Return LangChain StructuredTool instances bound to this registry's callbacks."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise ImportError(
                "LangChain support requires 'langchain-core'. "
                "Install with: pip install 'ecommerce-chat-agent[langchain]'"
            ) from exc

        async def _search(query: str) -> str:
            return await self.execute("search_products", {"query": query})

        async def _order(order_id: str) -> str:
            return await self.execute("get_order_status", {"order_id": order_id})

        return [
            StructuredTool.from_function(
                coroutine=_search,
                name="search_products",
                description=(
                    "Search the store's product catalog by keywords, product name, "
                    "category, or SKU."
                ),
                args_schema=SearchProductsInput,
            ),
            StructuredTool.from_function(
                coroutine=_order,
                name="get_order_status",
                description=(
                    "Look up the current fulfillment and shipping status of an order "
                    "by its order ID."
                ),
                args_schema=GetOrderStatusInput,
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Validate arguments and execute the requested tool callback."""
        try:
            if tool_name == "search_products":
                parsed = SearchProductsInput.model_validate(arguments)
                result = await self._search_products(parsed.query)
                return _serialize_tool_output(result)

            if tool_name == "get_order_status":
                parsed_order = GetOrderStatusInput.model_validate(arguments)
                result = await self._get_order_status(parsed_order.order_id)
                return _serialize_tool_output(result)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{tool_name}': {exc}"
            ) from exc
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Callback execution failed for tool '{tool_name}': {exc}"
            ) from exc

        raise ToolExecutionError(f"Unknown tool '{tool_name}'.")


def build_tool_registry(config: ChatConfig, vendor_id: str | None = None) -> ToolRegistry:
    """Construct a ToolRegistry for the active tier, injecting vendor_id for Enterprise."""
    validated_vendor = VendorIsolationGuard.validate_vendor_id(vendor_id, config.tier)

    if config.tier == PlanTier.ENTERPRISE:
        assert validated_vendor is not None
        search_cb, order_cb = VendorIsolationGuard.wrap_vendor_callbacks(
            config, validated_vendor
        )
        return ToolRegistry(
            search_products=search_cb,
            get_order_status=order_cb,
            vendor_id=validated_vendor,
        )

    assert config.search_products is not None and config.get_order_status is not None
    return ToolRegistry(
        search_products=config.search_products,
        get_order_status=config.get_order_status,
        vendor_id=None,
    )
