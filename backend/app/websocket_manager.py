"""WebSocket connection manager for real-time PlumeTrace telemetry."""

import asyncio
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Async-safe manager for connected frontend telemetry clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a frontend WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            count = len(self._connections)
        logger.info("WebSocket client connected. active_connections=%d", count)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection if it is currently tracked."""
        async with self._lock:
            self._connections.discard(websocket)
            count = len(self._connections)
        logger.info("WebSocket client disconnected. active_connections=%d", count)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast JSON telemetry to all connected clients."""
        async with self._lock:
            connections = tuple(self._connections)

        if not connections:
            return

        results = await asyncio.gather(
            *(self._safe_send(connection, message) for connection in connections),
            return_exceptions=True,
        )

        stale_connections = {
            connection
            for connection, result in zip(connections, results, strict=False)
            if result is False or isinstance(result, Exception)
        }
        if not stale_connections:
            return

        async with self._lock:
            self._connections.difference_update(stale_connections)
            count = len(self._connections)
        logger.info(
            "Removed stale WebSocket connections. removed=%d active_connections=%d",
            len(stale_connections),
            count,
        )

    async def _safe_send(self, websocket: WebSocket, message: dict[str, Any]) -> bool:
        """Send one payload and report whether the connection stayed healthy."""
        try:
            if websocket.application_state != WebSocketState.CONNECTED:
                return False
            await websocket.send_json(message)
            return True
        except Exception as exc:
            logger.warning("WebSocket send failed: %s", exc)
            return False

    async def close_all(self) -> None:
        """Close all tracked WebSocket connections during shutdown."""
        async with self._lock:
            connections = tuple(self._connections)
            self._connections.clear()

        for websocket in connections:
            try:
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1001, reason="Server shutdown")
            except Exception as exc:
                logger.debug("Ignoring WebSocket close failure during shutdown: %s", exc)


websocket_manager = WebSocketManager()
