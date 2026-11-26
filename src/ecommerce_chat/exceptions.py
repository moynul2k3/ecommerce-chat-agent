"""Custom exceptions for the ecommerce-chat-agent package."""


class EcommerceChatError(Exception):
    """Base exception for all ecommerce-chat-agent errors."""


class ConfigurationError(EcommerceChatError):
    """Raised when ChatConfig is invalid or missing required fields."""


class TierViolationError(EcommerceChatError):
    """Raised when an operation is not permitted under the configured PlanTier."""


class SessionNotFoundError(EcommerceChatError):
    """Raised when a requested chat session does not exist."""


class VendorIsolationError(EcommerceChatError):
    """Raised when a multi-vendor tenant boundary is missing or violated."""


class ToolExecutionError(EcommerceChatError):
    """Raised when an injected tool callback fails or an unknown tool is called."""


class AgentHandoffError(EcommerceChatError):
    """Raised when a human-agent handoff state transition is invalid."""
