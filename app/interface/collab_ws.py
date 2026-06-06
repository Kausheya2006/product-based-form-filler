"""
WebSocket-based real-time collaboration manager.
Each collaborative conversation session is identified by a room_id
(typically the conversation_id or a temporary UUID for new convos).
All connected users in the same room receive each other's messages instantly.
"""

import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Room manager ────────────────────────────────────────────────────────────

class CollabRoom:
    """Holds all active WebSocket connections for one session room."""

    def __init__(self, room_id: str):
        self.room_id = room_id
        # Maps username -> WebSocket
        self.connections: Dict[str, WebSocket] = {}

    @property
    def user_list(self) -> list[str]:
        return list(self.connections.keys())

    async def connect(self, username: str, ws: WebSocket):
        await ws.accept()
        self.connections[username] = ws
        logger.info(f"[COLLAB] {username} joined room {self.room_id}. Users: {self.user_list}")
        # Notify everyone that a new user joined
        await self.broadcast({
            "type": "user_joined",
            "username": username,
            "users": self.user_list,
        }, exclude=username)

    async def disconnect(self, username: str):
        self.connections.pop(username, None)
        logger.info(f"[COLLAB] {username} left room {self.room_id}. Users: {self.user_list}")
        await self.broadcast({
            "type": "user_left",
            "username": username,
            "users": self.user_list,
        })

    async def broadcast(self, payload: dict, exclude: str | None = None):
        """Send a JSON payload to all users in the room, optionally excluding one."""
        dead = []
        for uname, ws in self.connections.items():
            if uname == exclude:
                continue
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps(payload))
            except Exception as e:
                logger.warning(f"[COLLAB] Send failed for {uname}: {e}")
                dead.append(uname)
        for uname in dead:
            self.connections.pop(uname, None)

    async def send_to(self, username: str, payload: dict):
        """Send directly to one user."""
        ws = self.connections.get(username)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps(payload))

    def is_empty(self) -> bool:
        return len(self.connections) == 0


class RoomManager:
    """Global registry of all active rooms."""

    def __init__(self):
        self._rooms: Dict[str, CollabRoom] = {}

    def get_or_create(self, room_id: str) -> CollabRoom:
        if room_id not in self._rooms:
            self._rooms[room_id] = CollabRoom(room_id)
        return self._rooms[room_id]

    def cleanup(self, room_id: str):
        room = self._rooms.get(room_id)
        if room and room.is_empty():
            del self._rooms[room_id]
            logger.info(f"[COLLAB] Room {room_id} cleaned up (empty)")


manager = RoomManager()


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/collab/{room_id}")
async def collab_ws(
    websocket: WebSocket,
    room_id: str,
    username: str = Query(...),   # ?username=dr_smith
):
    """
    WebSocket endpoint for collaborative conversation entry.

    Clients connect to:  ws://host/ws/collab/{room_id}?username=alice

    Message types sent FROM client to server:
        { "type": "message",   "speaker": "Doctor", "text": "Hello" }
        { "type": "typing",    "speaker": "Doctor" }
        { "type": "ping" }

    Message types broadcast TO all clients in room:
        { "type": "message",   "speaker": "Doctor", "text": "Hello",
          "from": "alice",     "msg_id": "<uuid>" }
        { "type": "typing",    "speaker": "Doctor", "from": "alice" }
        { "type": "user_joined", "username": "alice", "users": [...] }
        { "type": "user_left",   "username": "alice", "users": [...] }
        { "type": "user_list",   "users": [...] }          (on connect)
        { "type": "pong" }
    """
    from uuid import uuid4

    room = manager.get_or_create(room_id)
    await room.connect(username, websocket)

    # Send the new user the current user list immediately
    await websocket.send_text(json.dumps({
        "type": "user_list",
        "users": room.user_list,
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "message":
                # Broadcast the new chat bubble to everyone including sender
                payload = {
                    "type": "message",
                    "from": username,
                    "speaker": data.get("speaker", username),
                    "text": data.get("text", ""),
                    "msg_id": str(uuid4()),
                }
                await room.broadcast(payload)   # includes sender so they get the msg_id
                logger.debug(f"[COLLAB] [{room_id}] {username}: {payload['text'][:60]}")

            elif msg_type == "typing":
                # Broadcast typing indicator to everyone else
                await room.broadcast({
                    "type": "typing",
                    "from": username,
                    "speaker": data.get("speaker", username),
                }, exclude=username)

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        await room.disconnect(username)
        manager.cleanup(room_id)