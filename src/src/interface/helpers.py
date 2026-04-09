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
from ..domain.speakers import render_history_for_model
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
    return templates.TemplateResponse(request, template_name, ctx)

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

def _is_global(obj) -> bool:
    """Return True if the object is a global/shared starter (owner_id is None)."""
    return getattr(obj, "owner_id", None) is None


def _can_write_form(form, user: dict) -> bool:
    """
    A user may EDIT or DELETE a form only if:
      - they are admin, OR
      - they own it (owner_id == user_id).
    Global forms (owner_id=None) are read-only for regular users.
    """
    if _is_admin(user):
        return True
    owner = getattr(form, "owner_id", None)
    return owner is not None and owner == user["user_id"]


def _can_write_convo(convo, user: dict) -> bool:
    """
    A user may EDIT or DELETE a conversation only if:
      - they are admin, OR
      - they own it (owner_id == user_id).
    Global conversations (owner_id=None) are read-only for regular users.
    """
    if _is_admin(user):
        return True
    owner = getattr(convo, "owner_id", None)
    return owner is not None and owner == user["user_id"]


async def _get_form_for_user(form_id: str, user: dict):
    """
    Return the form for READ access, or None if the user has no visibility.
    Regular users can see: global forms (owner_id=None) + their own forms.
    Admins can see everything.
    Use _can_write_form() separately before allowing edits or deletes.
    """
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
    """
    Return the conversation for READ access, or None if the user has no visibility.
    Regular users can see: global conversations (owner_id=None) + their own.
    Admins can see everything.
    Use _can_write_convo() separately before allowing edits or deletes.
    """
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

def _normalize_flat_field_map(payload: Dict[str, Any] | None) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for field_key, value in (payload or {}).items():
        if not isinstance(field_key, str):
            continue
        key = field_key.strip()
        if not key:
            continue
        normalized[key] = "" if value is None else str(value).strip()
    return normalized

def _flatten_nested_field_map(source: Dict[str, Any], prefix: str = "", out: Dict[str, str] | None = None) -> Dict[str, str]:
    if out is None:
        out = {}
    for key, value in (source or {}).items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_nested_field_map(value, dotted_key, out)
        else:
            out[dotted_key] = "" if value is None else str(value).strip()
    return out

def _merge_display_fields(
    filled_data: Dict[str, Any] | None,
    accepted_new_fields: Dict[str, Any] | None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(filled_data or {})
    for field_key, value in _normalize_flat_field_map(accepted_new_fields).items():
        _set_nested_field(merged, field_key, value)
    return merged

async def _extract_for_conversation_text(
    form: FormSchema,
    conversation_text: str,
    current_field_state: Dict[str, Any] | None = None,
    accepted_new_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed = _parse_conversation_text(conversation_text)
    full_convo = render_history_for_model(parsed)

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
    logger.info("[LiveExtract] Accepted new fields=%s", accepted_new_fields or {})
    logger.info("[LiveExtract] Seeded fields=%s", seeded_fields)
    logger.info("[LiveExtract] Parsed conversation=%s", parsed)
    logger.info("[LiveExtract] Model input=%s", input_str)

    model = container.pipeline.model
    if hasattr(model, "process_live_update"):
        answers_task = model.process_live_update(
            conversation_text=full_convo.strip(),
            form_name=form.name,
            current_field_state=seeded_fields,
            field_keys=list(form.fields.keys()),
            accepted_new_fields=_normalize_flat_field_map(accepted_new_fields),
        )
        logger.info("[LiveExtract] Using process_live_update path")
    else:
        answers_task = model.process_extraction_request(input_str)
        logger.info("[LiveExtract] Using full conversation replay path")
    summary_task = container.pipeline.summarizer.summarize(
        full_convo
    )
    answers, summary = await asyncio.gather(answers_task, summary_task)

    logger.info("[LiveExtract] Raw answers=%s", answers)
    logger.info("[LiveExtract] Summary=%s", summary)

    filled_data: Dict[str, Any] = {}
    suggested_new_fields: Dict[str, str] = {}

    if isinstance(answers, dict):
        model_filled = answers.get("filled_data", {})
        model_suggestions = answers.get("suggested_new_fields", {})
        flat_filled = _flatten_nested_field_map(model_filled) if isinstance(model_filled, dict) else {}
        flat_suggestions = _flatten_nested_field_map(model_suggestions) if isinstance(model_suggestions, dict) else {}

        for field_key in form.fields.keys():
            value = flat_filled.get(field_key, seeded_fields.get(field_key, "N/A"))
            logger.info("[LiveExtract] Assigning field %s -> %s", field_key, value)
            _set_nested_field(filled_data, field_key, value)

        for field_key, value in flat_suggestions.items():
            if field_key in form.fields:
                continue
            normalized_value = "" if value is None else str(value).strip()
            if not normalized_value:
                continue
            suggested_new_fields[field_key] = normalized_value
    else:
        for field_key, value in zip(form.fields.keys(), answers):
            logger.info("[LiveExtract] Assigning field %s -> %s", field_key, value)
            _set_nested_field(filled_data, field_key, value)
    for field_key in form.fields.keys():
        if not _has_nested_field(filled_data, field_key):
            logger.info("[LiveExtract] Missing field after assignment, defaulting %s -> N/A", field_key)
            _set_nested_field(filled_data, field_key, "N/A")

    normalized_accepted_new_fields = _normalize_flat_field_map(accepted_new_fields)
    if normalized_accepted_new_fields:
        filled_data = _merge_display_fields(filled_data, normalized_accepted_new_fields)

    logger.info("[LiveExtract] Final filled_data=%s", filled_data)
    logger.info("[LiveExtract] Suggested new fields=%s", suggested_new_fields)
    logger.info("[LiveExtract] Accepted new fields=%s", normalized_accepted_new_fields)

    return {
        "filled_data": filled_data,
        "suggested_new_fields": suggested_new_fields,
        "accepted_new_fields": normalized_accepted_new_fields,
        "summary": summary,
    }

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

async def _save_output(result: ExtractionResult, owner_id: str):
    collection = container.convo_repo.db.outputs
    doc = result.model_dump()
    doc["owner_id"] = owner_id
    await collection.insert_one(doc)
    logger.info("Extraction result saved to Atlas.")

async def _load_output_by_run_id(run_id: str, user: dict) -> Optional[Dict[str, Any]]:
    collection = container.convo_repo.db.outputs
    query: Dict[str, Any] = {"run_id": run_id}
    if not _is_admin(user):
        query["owner_id"] = user["user_id"]
    return await collection.find_one(query)

async def _load_outputs(user: dict) -> List[Dict]:
    collection = container.convo_repo.db.outputs
    if _is_admin(user):
        cursor = collection.find({}).sort("_id", -1)
    else:
        cursor = collection.find({"owner_id": user["user_id"]}).sort("_id", -1)
    return await cursor.to_list(length=100)
