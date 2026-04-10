from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.domain.domain import Conversation, ConversationVersion, FormSchema
from src.interface import api


class FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = list(docs)

    def sort(self, field: str, order: int):
        reverse = order == -1
        self._docs.sort(key=lambda d: d.get(field), reverse=reverse)
        return self

    async def to_list(self, length: int = 500):
        return list(self._docs)[:length]


class FakeUserRepo:
    def __init__(self):
        self.users: list[dict[str, Any]] = []

    async def create_index(self, *_args, **_kwargs):
        return None

    async def find_one(self, query: dict[str, Any]):
        for user in self.users:
            if all(user.get(k) == v for k, v in query.items()):
                return user
        return None

    async def insert_one(self, doc: dict[str, Any]):
        self.users.append(dict(doc))

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]):
        user = await self.find_one(query)
        if not user:
            return
        for key, value in update.get("$set", {}).items():
            user[key] = value

    async def delete_one(self, query: dict[str, Any]):
        for idx, user in enumerate(self.users):
            if all(user.get(k) == v for k, v in query.items()):
                del self.users[idx]
                return

    def find(self, query: dict[str, Any]):
        if not query:
            return FakeCursor(self.users)

        if "user_id" in query and isinstance(query["user_id"], dict):
            in_values = set(query["user_id"].get("$in", []))
            return FakeCursor([u for u in self.users if u.get("user_id") in in_values])

        return FakeCursor([u for u in self.users if all(u.get(k) == v for k, v in query.items())])


class FakeFormRepo:
    def __init__(self):
        self.forms: dict[str, FormSchema] = {}

    async def get_all(self):
        return list(self.forms.values())

    async def get_by_id(self, form_id: str):
        return self.forms.get(form_id)

    async def save(self, form: FormSchema):
        self.forms[form.id] = form

    async def delete_by_id(self, form_id: str):
        self.forms.pop(form_id, None)


class FakeConvoRepo:
    def __init__(self):
        self.conversations: dict[str, Conversation] = {}

    async def get_by_form_id(self, form_id: str):
        return [c for c in self.conversations.values() if c.form_id == form_id]

    async def get_by_id(self, convo_id: str):
        return self.conversations.get(convo_id)

    async def save(self, conversation: Conversation):
        self.conversations[conversation.id] = conversation


class FakeRunLogRepo:
    async def ensure_indexes(self):
        return None

    async def get_recent(self, _limit: int = 20):
        return []

    async def get_by_id(self, _run_id: str):
        return None

    async def update(self, _run_id: str, _data: dict[str, Any]):
        return None


@pytest.fixture
def test_state(monkeypatch: pytest.MonkeyPatch):
    users = FakeUserRepo()
    forms = FakeFormRepo()
    convos = FakeConvoRepo()

    api.container.convo_repo = convos
    api.container.form_repo = forms
    api.container.runlog_repo = FakeRunLogRepo()

    monkeypatch.setattr(api, "_user_repo", lambda: users)

    seeded_users: dict[str, dict[str, Any]] = {}

    async def _fake_get_current_user(request):
        key = request.headers.get("x-test-user", "")
        return seeded_users.get(key)

    monkeypatch.setattr(api, "_get_current_user", _fake_get_current_user)

    def add_user(*, key: str, username: str, password: str, role: str = "user"):
        user_doc = {
            "user_id": f"u-{key}",
            "username": username,
            "email": f"{username}@example.com",
            "password_hash": api._hash_password(password),
            "role": role,
            "created_at": datetime.utcnow(),
        }
        users.users.append(user_doc)
        seeded_users[key] = user_doc
        return user_doc

    def add_form(
        *,
        form_id: str,
        name: str,
        owner_id: str | None,
        visibility: str = "global",
        collaborators: list[str] | None = None,
    ):
        form = FormSchema(
            form_id=form_id,
            form_name=name,
            schema={"customer_name": "What is the customer name?"},
            owner_id=owner_id,
            visibility=visibility,
            collaborators=collaborators or [],
        )
        forms.forms[form_id] = form
        return form

    def add_conversation(
        *,
        convo_id: str,
        form_id: str,
        owner_id: str | None,
        name: str = "",
        history: dict[str, str] | None = None,
    ):
        convo = Conversation(
            conversation_id=convo_id,
            form_id=form_id,
            conversation_name=name,
            owner_id=owner_id,
            versions=[
                ConversationVersion(
                    version_index=0,
                    history=history or {"Speaker 1": "Hello there"},
                )
            ],
        )
        convos.conversations[convo_id] = convo
        return convo

    return {
        "user_repo": users,
        "form_repo": forms,
        "convo_repo": convos,
        "add_user": add_user,
        "add_form": add_form,
        "add_conversation": add_conversation,
    }


@pytest.fixture
def client():
    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    api.app.router.lifespan_context = _no_lifespan
    with TestClient(api.app) as test_client:
        yield test_client
