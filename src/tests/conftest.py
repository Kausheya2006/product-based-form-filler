from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.domain.domain import Conversation, ConversationVersion, FormSchema, ExtractionResult
from src.interface import helpers as interface_helpers

# api.py imports seed_data from helpers, so tests provide a no-op fallback
# without touching application code.
if not hasattr(interface_helpers, "seed_data"):
    interface_helpers.seed_data = lambda *args, **kwargs: None

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


class FakeOutputCollection:
    def __init__(self):
        self.docs: list[dict[str, Any]] = []

    async def insert_one(self, doc: dict[str, Any]):
        self.docs.append(dict(doc))

    async def find_one(self, query: dict[str, Any]):
        for doc in reversed(self.docs):
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def find(self, query: dict[str, Any]):
        if not query:
            return FakeCursor(list(self.docs))
        filtered = [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        return FakeCursor(filtered)


class FakeDb:
    def __init__(self):
        self.outputs = FakeOutputCollection()
        self.users = None


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
        self.db = FakeDb()

    async def get_by_form_id(self, form_id: str):
        return [c for c in self.conversations.values() if c.form_id == form_id]

    async def get_by_id(self, convo_id: str):
        return self.conversations.get(convo_id)

    async def save(self, conversation: Conversation):
        self.conversations[conversation.id] = conversation


class FakeRunLogRepo:
    def __init__(self):
        self.logs: dict[str, dict[str, Any]] = {}

    async def ensure_indexes(self):
        return None

    async def create(self, log):
        self.logs[log.run_id] = {
            "run_id": log.run_id,
            "conversation_id": log.conversation_id,
            "version_index": log.version_index,
            "owner_id": log.owner_id,
            "status": log.status,
            "summary": log.summary,
            "extracted_fields": dict(log.extracted_fields),
        }

    async def get_recent(self, _limit: int = 20):
        return list(self.logs.values())

    async def get_by_id(self, _run_id: str):
        data = self.logs.get(_run_id)
        if data is None:
            return None

        class _RunLog:
            pass

        log = _RunLog()
        for key, value in data.items():
            setattr(log, key, value)
        return log

    async def update(self, _run_id: str, _data: dict[str, Any]):
        existing = self.logs.get(_run_id, {"run_id": _run_id})
        existing.update(_data)
        self.logs[_run_id] = existing


class FakePipeline:
    class _Model:
        async def process_live_update(self, conversation_text, form_name, current_field_state, field_keys, accepted_new_fields=None):
            _ = (conversation_text, form_name, accepted_new_fields)
            return [current_field_state.get(k, "N/A") for k in field_keys]

    class _Summarizer:
        async def summarize(self, text):
            return f"Summary: {text[:80]}"

    def __init__(self, convo_repo: FakeConvoRepo, form_repo: FakeFormRepo):
        self._convo_repo = convo_repo
        self._form_repo = form_repo
        self._run_counter = 0
        self.model = self._Model()
        self.summarizer = self._Summarizer()

    async def run(self, conversation_id: str, form_id: str, version_index: int, owner_id: str | None = None):
        _ = (version_index, owner_id)
        convo = await self._convo_repo.get_by_id(conversation_id)
        form = await self._form_repo.get_by_id(form_id)
        if not convo or not form:
            raise ValueError("Conversation or form not found")

        self._run_counter += 1
        run_id = f"run-{self._run_counter}"
        filled = {field_name: "N/A" for field_name in form.fields.keys()}
        return ExtractionResult(
            conversation_id=conversation_id,
            form_id=form_id,
            filled_data=filled,
            run_id=run_id,
            summary="Stub summary for tests",
        )


@pytest.fixture
def test_state(monkeypatch: pytest.MonkeyPatch):
    users = FakeUserRepo()
    forms = FakeFormRepo()
    convos = FakeConvoRepo()
    convos.db.users = users
    runlogs = FakeRunLogRepo()

    api.container.convo_repo = convos
    api.container.form_repo = forms
    api.container.runlog_repo = runlogs
    api.container.pipeline = FakePipeline(convos, forms)

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
        "runlog_repo": runlogs,
        "outputs": convos.db.outputs,
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
