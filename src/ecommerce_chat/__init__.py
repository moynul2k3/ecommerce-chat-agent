"""Embeddable FastAPI e-commerce chat agent package."""

import utils.session
from ecommerce_chat.config import (
    ChatConfig,
    GetOrderStatusCallback,
    HandoffCallback,
    OrderStatusResult,
    PlanTier,
    ProductResult,
    SearchProductsCallback,
    VendorGetOrderStatusCallback,
    VendorSearchProductsCallback,
)
from ecommerce_chat.exceptions import (
    AgentHandoffError,
    ConfigurationError,
    EcommerceChatError,
    SessionNotFoundError,
    TierViolationError,
    ToolExecutionError,
    VendorIsolationError,
)
from ecommerce_chat.router import create_chat_router
from ecommerce_chat.sessions import (
    ChatMessage,
    ChatMode,
    InMemorySessionStore,
    RedisSessionStore,
    SessionState,
    SessionStore,
)
from ecommerce_chat.websocket import ConnectionManager

__all__ = [
    "AgentHandoffError",
    "ChatConfig",
    "ChatMessage",
    "ChatMode",
    "ConfigurationError",
    "ConnectionManager",
    "EcommerceChatError",
    "GetOrderStatusCallback",
    "HandoffCallback",
    "InMemorySessionStore",
    "OrderStatusResult",
    "PlanTier",
    "ProductResult",
    "RedisSessionStore",
    "SearchProductsCallback",
    "SessionNotFoundError",
    "SessionState",
    "SessionStore",
    "TierViolationError",
    "ToolExecutionError",
    "VendorGetOrderStatusCallback",
    "VendorIsolationError",
    "VendorSearchProductsCallback",
    "create_chat_router",
]

__version__ = "0.1.0"
