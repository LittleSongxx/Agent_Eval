"""In-process WebSocket connection manager for real-time task progress push.

评测任务在 daemon 工作线程中执行，该线程有自己的事件循环；WebSocket 连接
则挂在 FastAPI 的事件循环上。``send_json`` 检测到跨线程调用时，通过
``loop.call_soon_threadsafe`` 把发送任务切回服务端循环，保证线程安全。
"""

from __future__ import annotations

import asyncio
import logging
import typing as t

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[int, set[WebSocket]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the FastAPI event loop so background threads can push safely."""
        self._loop = loop

    async def connect(self, task_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.setdefault(task_id, set()).add(websocket)

    def disconnect(self, task_id: int, websocket: WebSocket) -> None:
        sockets = self.active_connections.get(task_id)
        if not sockets:
            return
        sockets.discard(websocket)
        if not sockets:
            self.active_connections.pop(task_id, None)

    def send_json(self, task_id: int, payload: dict[str, t.Any]) -> None:
        """Broadcast a payload to all clients of a task (thread-safe)."""
        sockets = list(self.active_connections.get(task_id) or [])
        if not sockets:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self._loop is not None and running is not self._loop:
            # 工作线程调用：把发送任务切回服务端事件循环
            try:
                self._loop.call_soon_threadsafe(
                    self._schedule_send, task_id, sockets, payload
                )
            except RuntimeError:
                logger.debug("Server loop closed; dropping WS push for task %s", task_id)
        else:
            asyncio.create_task(self._send_all(task_id, sockets, payload))

    def _schedule_send(
        self, task_id: int, sockets: list[WebSocket], payload: dict[str, t.Any]
    ) -> None:
        try:
            asyncio.get_running_loop().create_task(self._send_all(task_id, sockets, payload))
        except RuntimeError:
            logger.debug("No running loop available for WS push of task %s", task_id)

    async def _send_all(
        self, task_id: int, sockets: list[WebSocket], payload: dict[str, t.Any]
    ) -> None:
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception:
                self.disconnect(task_id, socket)


manager = ConnectionManager()
