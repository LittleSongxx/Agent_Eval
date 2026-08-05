"""WebSocket endpoints for real-time task progress push.

客户端连接后保持空闲即可收到服务端推送的 ``task_progress`` 消息；
前端在 WebSocket 断开时自动回退到轮询接口，保证进度不丢失。
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.ws_manager import manager

router = APIRouter()


@router.websocket("/ws/evaluations/{task_id}")
async def evaluation_progress_ws(websocket: WebSocket, task_id: int) -> None:
    await manager.connect(task_id, websocket)
    try:
        # 服务端单向推送，客户端无需发送内容；保持接收以感知连接断开
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(task_id, websocket)
