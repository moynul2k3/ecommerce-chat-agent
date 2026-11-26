"""WebSocket ConnectionManager handling customer and support agent rooms."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import WebSocket

from ecommerce_chat.config import PlanTier
from ecommerce_chat.exceptions import TierViolationError
from ecommerce_chat.multi_vendor.isolation import VendorIsolationGuard


class ConnectionManager:
    """Manages tenant-scoped WebSocket rooms for customers and human support agents."""

    def __init__(self, tier: PlanTier) -> None:
        self.tier = tier
        self._customer_rooms: dict[str, set[WebSocket]] = defaultdict(set)
        self._agent_rooms: dict[str, set[WebSocket]] = defaultdict(set)

    def _room_id(self, session_id: str, vendor_id: str | None) -> str:
        return VendorIsolationGuard.scope_room_id(session_id, vendor_id, self.tier)

    async def connect_customer(
        self,
        websocket: WebSocket,
        session_id: str,
        vendor_id: str | None = None,
    ) -> str:
        """Accept and register a customer WebSocket in the scoped session room."""
        room_id = self._room_id(session_id, vendor_id)
        await websocket.accept()
        self._customer_rooms[room_id].add(websocket)
        return room_id

    def disconnect_customer(
        self,
        websocket: WebSocket,
        session_id: str,
        vendor_id: str | None = None,
    ) -> None:
        """Unregister a customer WebSocket from the scoped session room."""
        room_id = self._room_id(session_id, vendor_id)
        sockets = self._customer_rooms.get(room_id)
        if sockets:
            sockets.discard(websocket)
            if not sockets:
                self._customer_rooms.pop(room_id, None)

    async def connect_agent(
        self,
        websocket: WebSocket,
        session_id: str,
        agent_id: str,
        vendor_id: str | None = None,
    ) -> str:
        """Accept and register a human support agent WebSocket (Pro & Enterprise only)."""
        if self.tier == PlanTier.FREE:
            raise TierViolationError(
                "Human support agent rooms are not available on the Free tier."
            )
        room_id = self._room_id(session_id, vendor_id)
        await websocket.accept()
        self._agent_rooms[room_id].add(websocket)
        return room_id

    def disconnect_agent(
        self,
        websocket: WebSocket,
        session_id: str,
        vendor_id: str | None = None,
    ) -> None:
        """Unregister a human support agent WebSocket from the scoped session room."""
        room_id = self._room_id(session_id, vendor_id)
        sockets = self._agent_rooms.get(room_id)
        if sockets:
            sockets.discard(websocket)
            if not sockets:
                self._agent_rooms.pop(room_id, None)

    @staticmethod
    async def _send_to_room(
        rooms: dict[str, set[WebSocket]],
        room_id: str,
        payload: dict[str, Any],
    ) -> int:
        sockets = list(rooms.get(room_id, ()))
        sent = 0
        for ws in sockets:
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception:
                room_set = rooms.get(room_id)
                if room_set:
                    room_set.discard(ws)
        return sent

    async def send_to_customer(
        self,
        session_id: str,
        payload: dict[str, Any],
        vendor_id: str | None = None,
    ) -> int:
        """Send a JSON payload to all customer sockets in the session room."""
        room_id = self._room_id(session_id, vendor_id)
        return await self._send_to_room(self._customer_rooms, room_id, payload)

    async def send_to_agent_room(
        self,
        session_id: str,
        payload: dict[str, Any],
        vendor_id: str | None = None,
    ) -> int:
        """Send a JSON payload to all human agent sockets in the session room."""
        room_id = self._room_id(session_id, vendor_id)
        return await self._send_to_room(self._agent_rooms, room_id, payload)

    async def broadcast_session(
        self,
        session_id: str,
        payload: dict[str, Any],
        vendor_id: str | None = None,
    ) -> int:
        """Send a JSON payload to both customer and agent sockets for a session."""
        customer_sent = await self.send_to_customer(
            session_id, payload, vendor_id=vendor_id
        )
        agent_sent = await self.send_to_agent_room(
            session_id, payload, vendor_id=vendor_id
        )
        return customer_sent + agent_sent

    def active_customer_count(
        self, session_id: str, vendor_id: str | None = None
    ) -> int:
        """Return the number of active customer sockets for a session."""
        room_id = self._room_id(session_id, vendor_id)
        return len(self._customer_rooms.get(room_id, ()))

    def active_agent_count(
        self, session_id: str, vendor_id: str | None = None
    ) -> int:
        """Return the number of active agent sockets for a session."""
        room_id = self._room_id(session_id, vendor_id)
        return len(self._agent_rooms.get(room_id, ()))
