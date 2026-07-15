"""Server-sent events connection manager for live telemetry fan-out."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

TelemetryQueue = asyncio.Queue[str]


class SSEManager:
    """Track active SSE queues and broadcast telemetry without blocking ingestion."""

    def __init__(self) -> None:
        self._queues: set[TelemetryQueue] = set()
        self._lock = asyncio.Lock()

    async def add_queue(self, queue: TelemetryQueue) -> None:
        async with self._lock:
            self._queues.add(queue)
        logger.debug("SSE client connected; active_clients=%d", await self.connection_count())

    async def remove_queue(self, queue: TelemetryQueue) -> None:
        async with self._lock:
            self._queues.discard(queue)
        logger.debug("SSE client disconnected; active_clients=%d", await self.connection_count())

    async def connection_count(self) -> int:
        async with self._lock:
            return len(self._queues)

    async def broadcast(self, message: str) -> None:
        """Fan out a JSON telemetry payload to all active clients.

        Slow clients are detached once their bounded queue fills. This keeps MQTT
        ingestion and database writes healthy even if a browser tab stops reading.
        """
        async with self._lock:
            queues = tuple(self._queues)

        stale_queues: list[TelemetryQueue] = []
        for queue in queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                stale_queues.append(queue)
            except Exception:
                # Queue in unexpected state — detach it
                stale_queues.append(queue)

        if stale_queues:
            async with self._lock:
                for queue in stale_queues:
                    self._queues.discard(queue)
            logger.warning("Dropped %d slow SSE client(s).", len(stale_queues))


sse_manager = SSEManager()
