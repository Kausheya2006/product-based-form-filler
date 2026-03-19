import os
import re
import json
import time
import hashlib
import secrets
import logging
import asyncio

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request
from fastapi.templating import Jinja2Templates

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..domain.domain import Conversation, FormSchema, ExtractionResult, ConversationVersion
from .dependencies import container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="src/interface/templates")

SECRET_KEY = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_signer = URLSafeTimedSerializer(SECRET_KEY)
SESSION_COOKIE = "pl_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 14  # 14 days

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "PLadmin")

def _hash_password(password: str) -> str:
    """SHA-256 hash with a random salt, stored as 'salt$hash'."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"

def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split("$", 1)
    except ValueError:
        return False
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

def _make_session_token(user_id: str) -> str:
    return _signer.dumps(user_id)

def _read_session_token(token: str) -> Optional[str]:
    try:
        return _signer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None

def _user_repo():
    """Return the MongoDB 'users' collection."""
    return container.convo_repo.db.users

def _is_admin(user: dict) -> bool:
    """Return True if the user document carries the admin role."""
    return user.get("role") == "admin"

async def _get_current_user(request: Request) -> Optional[dict]:
    """Resolve the session cookie to a user document, or return None."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = _read_session_token(token)
    if not user_id:
        return None
    return await _user_repo().find_one({"user_id": user_id})

def _tmpl(template_name: str, request: Request, ctx: dict, user=None):
    """Render a template, always injecting current_user and is_admin."""
    ctx.setdefault("request", request)
    ctx["current_user"] = user
    ctx["is_admin"] = _is_admin(user) if user else False
    return templates.TemplateResponse(template_name, ctx)

def _validate_username(username: str) -> Optional[str]:
    """
    Validate a username string.
    Returns an error message string on failure, or None if valid.
    """
    if len(username) < 3 or len(username) > 32:
        return "Username must be 3-32 characters."
    if not re.match(r'^[a-zA-Z0-9_\-]+$', username):
        return "Username may only contain letters, numbers, underscores, and hyphens."
    return None

def _default_field_question(field_name: str) -> str:
    pretty = field_name.replace(".", " ").replace("_", " ").strip()
    pretty = " ".join(pretty.split())
    if not pretty:
        return "What value should be captured for this field?"
    starts_like_question = (
        "what", "when", "where", "who", "which", "why", "how",
        "is", "are", "do", "does", "did", "can", "could", "should",
        "would", "will", "has", "have", "had"
    )
    lowered = pretty.lower()
    if lowered.startswith(starts_like_question):
        return pretty if pretty.endswith("?") else f"{pretty}?"
    return f"What is the {lowered}?"

def _build_schema_from_pairs(
    field_names: List[str],
    field_values: List[str],
    *,
    autogenerate_question: bool = False,
) -> Dict[str, str]:
    schema: Dict[str, str] = {}
    for raw_name, raw_value in zip(field_names, field_values):
        name = raw_name.strip()
        if not name:
            continue
        value = (raw_value or "").strip()
        if not value and autogenerate_question:
            value = _default_field_question(name)
        if not value:
            continue
        schema[name] = value
    return schema

def _convert_mongo_types(obj):
    if isinstance(obj, dict):
        if "$date" in obj:
            s = obj["$date"]
            if isinstance(s, dict) and "$numberLong" in s:
                s = s["$numberLong"]
            if isinstance(s, str):
                if s.endswith("Z"):
                    s = s.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(s)
                except Exception:
                    return s
        if "$oid" in obj:
            return str(obj["$oid"])
        return {k: _convert_mongo_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_mongo_types(i) for i in obj]
    return obj

async def seed_data():
    """Seed MongoDB from JSON files (global/shared starter data)."""
    if os.path.exists("data/conversations.json"):
        with open("data/conversations.json", "r") as f:
            for raw in json.load(f):
                data = _convert_mongo_types(raw)
                data["conversation_id"] = str(data.get("conversation_id", ""))
                history = data.pop("history", data.pop("conversation", None))
                if history and not data.get("versions"):
                    data["versions"] = [
                        ConversationVersion(version_index=0, history=history).model_dump()
                    ]
                data.setdefault("owner_id", None)
                await container.convo_repo.save(Conversation(**data))
        logger.info("Conversations seeded.")
    if os.path.exists("data/forms.json"):
        with open("data/forms.json", "r") as f:
            for raw in json.load(f):
                data = _convert_mongo_types(raw)
                data["form_id"] = str(data.get("form_id", ""))
                data.setdefault("owner_id", None)
                await container.form_repo.save(FormSchema(**data))
        logger.info("Forms seeded.")

async def _get_form_for_user(form_id: str, user: dict):
    """Admin sees all. Others see global (owner_id=None) or their own."""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        return None
    if _is_admin(user):
        return form
    owner = getattr(form, "owner_id", None)
    if owner is None or owner == user["user_id"]:
        return form
    return None

async def _get_convo_for_user(convo_id: str, user: dict):
    """Admin sees all. Others see global (owner_id=None) or their own."""
    convo = await container.convo_repo.get_by_id(convo_id)
    if not convo:
        return None
    if _is_admin(user):
        return convo
    owner = getattr(convo, "owner_id", None)
    if owner is None or owner == user["user_id"]:
        return convo
    return None

def _parse_conversation_text(text: str) -> Dict[str, str]:
    result = {}
    lines = text.strip().split("\n")
    base_timestamp = int(time.time())
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            result[f"{speaker} {base_timestamp + i}"] = message
    return result

def _set_nested_field(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    if "." not in dotted_key:
        target[dotted_key] = value
        return
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

def _has_nested_field(source: Dict[str, Any], dotted_key: str) -> bool:
    parts = dotted_key.split(".")
    current: Any = source
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return isinstance(current, dict) and parts[-1] in current

def _apply_field_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for field_key, value in overrides.items():
        if not isinstance(field_key, str) or not field_key.strip():
            continue
        _set_nested_field(target, field_key, "" if value is None else str(value))

async def _extract_for_conversation_text(
    form: FormSchema,
    conversation_text: str,
    current_field_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed = _parse_conversation_text(conversation_text)
    full_convo = ""
    for speaker, text in parsed.items():
        clean_speaker = " ".join(speaker.split()[:-1]) if " " in speaker else speaker
        full_convo += f"{clean_speaker}: {text}\n"

    seeded_fields: Dict[str, Any] = {}
    for field_key in form.fields.keys():
        if current_field_state and field_key in current_field_state:
            candidate = current_field_state[field_key]
            normalized = "" if candidate is None else str(candidate).strip()
            seeded_fields[field_key] = normalized if normalized else "N/A"
        else:
            seeded_fields[field_key] = "N/A"

    input_str = (
        f"Extract info from conversation to fill form.\n"
        f"Conversation: {full_convo}"
        f"Form: {form.name}\n"
        f"Fields: {json.dumps(seeded_fields)}"
    )

    logger.info("[LiveExtract] Form=%s", form.name)
    logger.info("[LiveExtract] Form fields=%s", list(form.fields.keys()))
    logger.info("[LiveExtract] Current field state=%s", current_field_state or {})
    logger.info("[LiveExtract] Seeded fields=%s", seeded_fields)
    logger.info("[LiveExtract] Parsed conversation=%s", parsed)
    logger.info("[LiveExtract] Model input=%s", input_str)

    answers_task = container.pipeline.model.process_extraction_request(input_str)
    summary_task = container.pipeline.summarizer.summarize(
        "\n".join([f"{k}: {v}" for k, v in parsed.items()])
    )
    answers, summary = await asyncio.gather(answers_task, summary_task)

    logger.info("[LiveExtract] Raw answers=%s", answers)
    logger.info("[LiveExtract] Summary=%s", summary)

    filled_data: Dict[str, Any] = {}
    for field_key, value in zip(form.fields.keys(), answers):
        logger.info("[LiveExtract] Assigning field %s -> %s", field_key, value)
        _set_nested_field(filled_data, field_key, value)
    for field_key in form.fields.keys():
        if not _has_nested_field(filled_data, field_key):
            logger.info("[LiveExtract] Missing field after assignment, defaulting %s -> N/A", field_key)
            _set_nested_field(filled_data, field_key, "N/A")

    logger.info("[LiveExtract] Final filled_data=%s", filled_data)

    return {"filled_data": filled_data, "summary": summary}

def _format_filled_data(data: Dict[str, Any], prefix: str = "") -> str:
    html = ""
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            html += _format_filled_data(v, key)
        else:
            html += (
                f'<div class="field">'
                f'<span class="field-name">{key}:</span> '
                f'<span class="field-value">{v}</span>'
                f'</div>'
            )
    return html

async def _save_output(result: ExtractionResult, owner_id: str = None):
    collection = container.convo_repo.db.outputs
    doc = result.model_dump()
    if owner_id:
        doc["owner_id"] = owner_id
    await collection.insert_one(doc)
    logger.info("Extraction result saved to Atlas.")

async def _load_outputs(user: dict) -> List[Dict]:
    collection = container.convo_repo.db.outputs
    if _is_admin(user):
        cursor = collection.find({}).sort("_id", -1)
    else:
        cursor = collection.find({
            "$or": [
                {"owner_id": user["user_id"]},
                {"owner_id": {"$exists": False}},
                {"owner_id": None},
            ]
        }).sort("_id", -1)
    return await cursor.to_list(length=100)
