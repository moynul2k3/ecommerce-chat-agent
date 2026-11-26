"""Configuration schemas, plan tiers, and injected callback contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecommerce_chat.exceptions import ConfigurationError, TierViolationError


class PlanTier(str, Enum):
    """Supported subscription tiers controlling available features."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ProductResult(BaseModel):
    """Standardized product representation returned by catalog search callbacks."""

    id: str
    name: str
    price: float
    currency: str = "USD"
    description: str = ""
    in_stock: bool = True
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderStatusResult(BaseModel):
    """Standardized order status representation returned by order lookup callbacks."""

    order_id: str
    status: str
    updated_at: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


SearchProductsCallback = Callable[
    [str], Awaitable[list[dict[str, Any]] | list[ProductResult]]
]
GetOrderStatusCallback = Callable[
    [str], Awaitable[dict[str, Any] | OrderStatusResult | None]
]

VendorSearchProductsCallback = Callable[
    [str, str], Awaitable[list[dict[str, Any]] | list[ProductResult]]
]
VendorGetOrderStatusCallback = Callable[
    [str, str], Awaitable[dict[str, Any] | OrderStatusResult | None]
]

HandoffCallback = Callable[[str, str | None, str | None], Awaitable[None]]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant. "
    "Use the provided tools to search for products and check order statuses. "
    "Be concise, accurate, and polite."
)


class ChatConfig(BaseModel):
    """Runtime configuration for creating an embeddable e-commerce chat router."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: PlanTier = PlanTier.FREE

    # Free & Pro single-tenant callbacks
    search_products: SearchProductsCallback | None = None
    get_order_status: GetOrderStatusCallback | None = None

    # Enterprise multi-vendor callbacks
    vendor_search_products: VendorSearchProductsCallback | None = None
    vendor_get_order_status: VendorGetOrderStatusCallback | None = None

    # Redis configuration (required for Pro and Enterprise)
    redis_url: str | None = None
    redis_client: Any | None = None
    session_ttl_seconds: int = Field(default=86400, ge=60)

    # LLM configuration
    openai_api_key: str | None = None
    model_name: str = "gpt-4o-mini"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tool_iterations: int = Field(default=5, ge=1, le=20)
    llm_agent: Any | None = None

    # Handoff & command configuration
    escalation_commands: tuple[str, ...] = ("/human", "/agent", "/escalate")
    return_to_ai_commands: tuple[str, ...] = ("/ai", "/bot", "/resolve")
    on_handoff: HandoffCallback | None = None

    @property
    def supports_human_handoff(self) -> bool:
        """Return True if the configured tier supports live human handoff."""
        return self.tier in (PlanTier.PRO, PlanTier.ENTERPRISE)

    @property
    def is_multi_vendor(self) -> bool:
        """Return True if the configured tier enforces multi-vendor isolation."""
        return self.tier == PlanTier.ENTERPRISE

    @model_validator(mode="after")
    def _validate_tier_requirements(self) -> ChatConfig:
        if self.tier == PlanTier.ENTERPRISE:
            if not self.vendor_search_products or not self.vendor_get_order_status:
                raise ConfigurationError(
                    "Enterprise tier requires both 'vendor_search_products' and "
                    "'vendor_get_order_status' callbacks."
                )
        else:
            if self.vendor_search_products or self.vendor_get_order_status:
                raise TierViolationError(
                    "Multi-vendor callbacks ('vendor_search_products', "
                    "'vendor_get_order_status') are only available in the Enterprise tier."
                )
            if not self.search_products or not self.get_order_status:
                raise ConfigurationError(
                    f"{self.tier.value.title()} tier requires both 'search_products' "
                    "and 'get_order_status' callbacks."
                )

        if self.tier in (PlanTier.PRO, PlanTier.ENTERPRISE):
            if not self.redis_url and self.redis_client is None:
                raise ConfigurationError(
                    f"{self.tier.value.title()} tier requires 'redis_url' or 'redis_client'."
                )

        return self
