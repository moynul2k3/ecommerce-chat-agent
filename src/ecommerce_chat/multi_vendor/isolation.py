"""Multi-tenant scoping and guard validation for Enterprise deployments."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, field_validator

from ecommerce_chat.config import (
    ChatConfig,
    GetOrderStatusCallback,
    PlanTier,
    SearchProductsCallback,
)
from ecommerce_chat.exceptions import TierViolationError, VendorIsolationError

_VENDOR_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class VendorContext(BaseModel):
    """Validated multi-vendor execution context."""

    vendor_id: str

    @field_validator("vendor_id")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _VENDOR_ID_PATTERN.match(cleaned):
            raise VendorIsolationError(
                "Invalid vendor_id: must be 1-128 alphanumeric, '.', '_', or '-' characters."
            )
        return cleaned


class VendorIsolationGuard:
    """Enforces strict tenant boundaries across sessions, WebSockets, and tools."""

    @staticmethod
    def validate_vendor_id(vendor_id: str | None, tier: PlanTier) -> str | None:
        """Validate vendor_id presence and format according to the active PlanTier."""
        if tier == PlanTier.ENTERPRISE:
            if vendor_id is None or not vendor_id.strip():
                raise VendorIsolationError(
                    "Enterprise tier requires a non-empty 'vendor_id' on every request."
                )
            return VendorContext(vendor_id=vendor_id).vendor_id

        if vendor_id is not None:
            raise TierViolationError(
                "Multi-vendor 'vendor_id' scoping is only available in the Enterprise tier."
            )
        return None

    @staticmethod
    def assert_session_vendor(
        session_vendor_id: str | None,
        request_vendor_id: str | None,
        tier: PlanTier,
    ) -> None:
        """Verify that a loaded session belongs to the requesting vendor."""
        if tier != PlanTier.ENTERPRISE:
            return
        if not request_vendor_id or session_vendor_id != request_vendor_id:
            raise VendorIsolationError(
                f"Session vendor mismatch: expected '{request_vendor_id}', "
                f"found '{session_vendor_id}'."
            )

    @staticmethod
    def scope_session_key(
        session_id: str,
        vendor_id: str | None,
        tier: PlanTier,
    ) -> str:
        """Build a tenant-isolated Redis key for a session."""
        validated_vendor = VendorIsolationGuard.validate_vendor_id(vendor_id, tier)
        if tier == PlanTier.ENTERPRISE:
            return f"ecommerce_chat:vendor:{validated_vendor}:session:{session_id}"
        return f"ecommerce_chat:session:{session_id}"

    @staticmethod
    def scope_room_id(
        session_id: str,
        vendor_id: str | None,
        tier: PlanTier,
    ) -> str:
        """Build a tenant-isolated WebSocket room identifier."""
        validated_vendor = VendorIsolationGuard.validate_vendor_id(vendor_id, tier)
        if tier == PlanTier.ENTERPRISE:
            return f"vendor:{validated_vendor}:session:{session_id}"
        return f"session:{session_id}"

    @staticmethod
    def wrap_vendor_callbacks(
        config: ChatConfig,
        vendor_id: str,
    ) -> tuple[SearchProductsCallback, GetOrderStatusCallback]:
        """Bind a validated vendor_id into Enterprise callbacks so LLM tools cannot spoof it."""
        validated_vendor = VendorIsolationGuard.validate_vendor_id(
            vendor_id, PlanTier.ENTERPRISE
        )
        assert validated_vendor is not None
        if not config.vendor_search_products or not config.vendor_get_order_status:
            raise VendorIsolationError("Enterprise vendor callbacks are not configured.")

        vendor_search = config.vendor_search_products
        vendor_order = config.vendor_get_order_status

        async def scoped_search_products(query: str) -> Any:
            return await vendor_search(validated_vendor, query)

        async def scoped_get_order_status(order_id: str) -> Any:
            return await vendor_order(validated_vendor, order_id)

        return scoped_search_products, scoped_get_order_status
