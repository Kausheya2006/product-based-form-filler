from datetime import datetime
from typing import Any
from uuid import uuid4
import os

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient
from pymongo.database import Database

from src.domain.domain import Conversation, ConversationVersion, FormSchema
from src.infrastructure.config import settings
from src.interface import api
from src.interface import helpers as interface_helpers

# api.py imports seed_data from helpers, so tests provide a no-op fallback
# without touching application code.
if not hasattr(interface_helpers, "seed_data"):
    interface_helpers.seed_data = lambda *args, **kwargs: None


class _FormsView:
    def __init__(self, db: Database):
        self._db = db

    def values(self):
        docs = list(self._db.forms.find({}, {"_id": 0}))
        return [FormSchema(**doc) for doc in docs]

    def __contains__(self, form_id: str) -> bool:
        return self._db.forms.count_documents({"form_id": form_id}, limit=1) > 0


class _FormRepoView:
    def __init__(self, db: Database):
        self.forms = _FormsView(db)


class _UserRepoView:
    def __init__(self, db: Database):
        self._db = db

    @property
    def users(self) -> list[dict[str, Any]]:
        return list(self._db.users.find({}, {"_id": 0}))


class _OutputDocsView:
    def __init__(self, db: Database):
        self._db = db

    def append(self, doc: dict[str, Any]) -> None:
        self._db.outputs.insert_one(dict(doc))

    def __getitem__(self, index: int) -> dict[str, Any]:
        docs = list(self._db.outputs.find({}).sort("_id", 1))
        if not docs:
            raise IndexError("No output docs present")
        doc = dict(docs[index])
        doc.pop("_id", None)
        return doc


class _OutputsView:
    def __init__(self, db: Database):
        self.docs = _OutputDocsView(db)


class _ConversationRef:
    def __init__(self, db: Database, conversation_id: str):
        self._db = db
        self.id = conversation_id

    @property
    def versions(self) -> list[ConversationVersion]:
        doc = self._db.conversations.find_one({"conversation_id": self.id}, {"_id": 0})
        if not doc:
            return []
        convo = Conversation(**doc)
        return convo.versions


@pytest.fixture(scope="session", autouse=True)
def configure_test_settings():
    settings.MONGO_URI = "mongodb://localhost:27017"
    settings.DB_NAME = "chat_db_pytest"
    settings.MOCK_MODELS = os.getenv("MOCK_MODELS", "false").lower() == "true"
    settings.USE_MODAL_INFERENCE = False
    settings.USE_LOCAL_CONTAINER_GEMMA4 = False
    settings.USE_OLLAMA = False
    settings.MODEL_SERVICE_URL = ""


@pytest.fixture(scope="session")
def mongo_db_session() -> Database:
    client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        pytest.skip(f"MongoDB is required for integration tests: {exc}")

    db = client[settings.DB_NAME]

    yield db

    client.close()


@pytest.fixture
def mongo_db(mongo_db_session: Database) -> Database:
    db = mongo_db_session
    db.users.delete_many({})
    db.forms.delete_many({})
    db.conversations.delete_many({})
    db.outputs.delete_many({})
    db.run_logs.delete_many({})

    yield db

    db.users.delete_many({})
    db.forms.delete_many({})
    db.conversations.delete_many({})
    db.outputs.delete_many({})
    db.run_logs.delete_many({})


@pytest.fixture(scope="session")
def client(mongo_db_session: Database):
    # Depend on Mongo availability; app lifespan (and model load) runs once per session.
    _ = mongo_db_session
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def test_state(client: TestClient, mongo_db: Database):
    users = mongo_db.users
    forms = mongo_db.forms
    conversations = mongo_db.conversations
    credentials_by_user_id: dict[str, str] = {}

    def _login_cookie(user_doc: dict[str, Any]) -> dict[str, str]:
        password = credentials_by_user_id[user_doc["user_id"]]
        response = client.post(
            "/login",
            data={"username": user_doc["username"], "password": password},
            follow_redirects=False,
        )
        assert response.status_code == 303
        token = response.cookies.get(api.SESSION_COOKIE)
        assert token
        return {api.SESSION_COOKIE: token}

    def add_user(*, key: str, username: str, password: str, role: str = "user"):
        email = f"{username}.{key}.{uuid4().hex[:6]}@example.com"
        response = client.post(
            "/register",
            data={
                "email": email,
                "username": username,
                "password": password,
                "confirm_password": password,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        if role != "user":
            users.update_one({"username": username}, {"$set": {"role": role}})

        user_doc = users.find_one({"username": username}, {"_id": 0})
        assert user_doc is not None
        credentials_by_user_id[user_doc["user_id"]] = password
        user_doc["__test_password"] = password
        return user_doc

    def add_form(
        *,
        form_id: str,
        name: str,
        owner_id: str | None,
        visibility: str = "global",
        collaborators: list[str] | None = None,
    ):
        _ = form_id
        owner_doc = users.find_one({"user_id": owner_id}, {"_id": 0})
        assert owner_doc is not None

        response = client.post(
            "/forms",
            data={
                "form_name": name,
                "form_description": f"{name} description",
                "visibility": visibility,
                "field_name[]": ["customer_name"],
                "field_type[]": ["Name"],
                "collaborator[]": collaborators or [],
            },
            cookies=_login_cookie(owner_doc),
            follow_redirects=False,
        )
        assert response.status_code == 303

        location = response.headers.get("location", "")
        created_form_id = ""
        if location.startswith("/forms/"):
            created_form_id = location.split("/forms/", 1)[1].split("?", 1)[0]

        if not created_form_id:
            latest = forms.find_one(
                {"owner_id": owner_id, "form_name": name},
                sort=[("_id", -1)],
            )
            assert latest is not None
            created_form_id = latest["form_id"]

        form_doc = forms.find_one({"form_id": created_form_id}, {"_id": 0})
        assert form_doc is not None
        form = FormSchema(**form_doc)
        return form

    def add_conversation(
        *,
        convo_id: str,
        form_id: str,
        owner_id: str | None,
        name: str = "",
        history: dict[str, str] | None = None,
    ):
        owner_doc = users.find_one({"user_id": owner_id}, {"_id": 0})
        assert owner_doc is not None

        seed_history = history or {"Speaker 1": "Hello there"}
        conversation_text = "\n".join(f"{speaker}: {text}" for speaker, text in seed_history.items())

        response = client.post(
            "/conversations/create",
            data={
                "form_id": form_id,
                "conversation_id": convo_id,
                "conversation_name": name,
                "conversation_text": conversation_text,
                "extract": "false",
            },
            cookies=_login_cookie(owner_doc),
            follow_redirects=False,
        )
        assert response.status_code == 303

        convo_doc = conversations.find_one({"conversation_id": convo_id}, {"_id": 0})
        assert convo_doc is not None
        return _ConversationRef(mongo_db, convo_id)

    return {
        "user_repo": _UserRepoView(mongo_db),
        "form_repo": _FormRepoView(mongo_db),
        "outputs": _OutputsView(mongo_db),
        "add_user": add_user,
        "add_form": add_form,
        "add_conversation": add_conversation,
    }
