import os
import re
import json
import time
import hashlib
import secrets
import logging
import asyncio

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from uuid import uuid4

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..domain.domain import Conversation, FormSchema, ExtractionResult, ConversationVersion
from .model_service_api import LiveExtractRequest
from ..domain.speakers import render_history_for_model
from .dependencies import container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="src/interface/templates")

SECRET_KEY      = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
SESSION_COOKIE  = "pl_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 14  # 14 days
ADMIN_USERNAME  = os.environ.get("ADMIN_USERNAME", "PLadmin")

SUPPORTED_INPUT_LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
]

def _transcript_to_conversation_text(transcript: str) -> str:
    cleaned = " ".join((transcript or "").split())
    if not cleaned:
        return ""
    chunks: list[str] = []
    sentence_buffer   = ""
    for token in cleaned.split(" "):
        candidate = f"{sentence_buffer} {token}".strip() if sentence_buffer else token
        if len(candidate) >= 280 and sentence_buffer:
            chunks.append(sentence_buffer.strip())
            sentence_buffer = token
        else:
            sentence_buffer = candidate
    if sentence_buffer.strip():
        chunks.append(sentence_buffer.strip())
    return "\n".join(f"Speaker: {chunk}" for chunk in chunks if chunk)

_signer = URLSafeTimedSerializer(SECRET_KEY)

class AuthService:
    """Handles password hashing/verification and session-token lifecycle."""

    @staticmethod
    def hash_password(password: str) -> str:
        """SHA-256 hash with a random salt, stored as 'salt$hash'."""
        salt   = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}${hashed}"

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        try:
            salt, hashed = stored.split("$", 1)
        except ValueError:
            return False
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

    @staticmethod
    def make_session_token(user_id: str) -> str:
        return _signer.dumps(user_id)

    @staticmethod
    def read_session_token(token: str) -> Optional[str]:
        try:
            return _signer.loads(token, max_age=SESSION_MAX_AGE)
        except (BadSignature, SignatureExpired):
            return None

    @staticmethod
    async def get_current_user(request: Request) -> Optional[dict]:
        """Resolve the session cookie to a user document, or return None."""
        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            return None
        user_id = AuthService.read_session_token(token)
        if not user_id:
            return None
        return await UserRepository.find_by_id(user_id)

class UserRepository:
    """Thin wrapper around the MongoDB 'users' collection."""

    @staticmethod
    def collection():
        return container.convo_repo.db.users

    @staticmethod
    async def find_by_id(user_id: str) -> Optional[dict]:
        return await UserRepository.collection().find_one({"user_id": user_id})

    @staticmethod
    async def find_by_username(username: str) -> Optional[dict]:
        return await UserRepository.collection().find_one({"username": username})

    @staticmethod
    async def find_by_email(email: str) -> Optional[dict]:
        return await UserRepository.collection().find_one({"email": email.lower()})

    @staticmethod
    async def insert(doc: dict) -> None:
        await UserRepository.collection().insert_one(doc)

    @staticmethod
    async def update(query: dict, update: dict) -> None:
        await UserRepository.collection().update_one(query, update)

    @staticmethod
    async def delete(user_id: str) -> None:
        await UserRepository.collection().delete_one({"user_id": user_id})

    @staticmethod
    async def list_all(limit: int = 500) -> List[dict]:
        return await UserRepository.collection().find({}).sort("created_at", 1).to_list(length=limit)

    @staticmethod
    async def find_by_ids(user_ids: List[str]) -> List[dict]:
        return await UserRepository.collection().find(
            {"user_id": {"$in": user_ids}}
        ).to_list(length=500)

    @staticmethod
    async def validate_username(username: str) -> Optional[str]:
        """Return an error string if invalid, else None."""
        if len(username) < 3 or len(username) > 32:
            return "Username must be 3-32 characters."
        if not re.match(r'^[a-zA-Z0-9_\-]+$', username):
            return "Username may only contain letters, numbers, underscores, and hyphens."
        return None
    
class AccessPolicy:
    """Centralises all permission logic."""

    @staticmethod
    def is_admin(user: dict) -> bool:
        return user.get("role") == "admin"

    @staticmethod
    def is_global(obj) -> bool:
        """True when owner_id is None (shared/seed data)."""
        return getattr(obj, "owner_id", None) is None

    @staticmethod
    def can_write_form(form, user: dict) -> bool:
        """Admin always yes; regular user only if they own the form."""
        if AccessPolicy.is_admin(user):
            return True
        owner = getattr(form, "owner_id", None)
        return owner is not None and owner == user["user_id"]

    @staticmethod
    def can_write_convo(convo, user: dict) -> bool:
        """Admin always yes; regular user only if they own the conversation."""
        if AccessPolicy.is_admin(user):
            return True
        owner = getattr(convo, "owner_id", None)
        return owner is not None and owner == user["user_id"]

    @staticmethod
    def can_read_form(form, user: dict) -> bool:
        """Global forms (owner_id=None), own forms, or collaborative membership."""
        if AccessPolicy.is_admin(user):
            return True
        owner = getattr(form, "owner_id", None)
        if owner is None or owner == user["user_id"]:
            return True
        if getattr(form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(form, "collaborators", []):
            return True
        return False

    @staticmethod
    def can_read_convo(convo, user: dict) -> bool:
        if AccessPolicy.is_admin(user):
            return True
        owner = getattr(convo, "owner_id", None)
        return owner is None or owner == user["user_id"]

class FormQueryService:
    """Read-access helpers for FormSchema objects."""

    @staticmethod
    async def get_for_user(form_id: str, user: dict) -> Optional[FormSchema]:
        """Return the form if the user may read it, else None."""
        form = await container.form_repo.get_by_id(form_id)
        if not form:
            return None
        if AccessPolicy.can_read_form(form, user):
            return form
        return None

    @staticmethod
    async def get_all_visible(user: dict) -> List[FormSchema]:
        all_forms = await container.form_repo.get_all()
        if AccessPolicy.is_admin(user):
            return all_forms
        username = user["username"]
        return [
            f for f in all_forms
            if getattr(f, "visibility", None) == "global"
            or getattr(f, "owner_id", None) is None
            or getattr(f, "owner_id", None) == user["user_id"]
            or (
                getattr(f, "visibility", "") == "collaborative"
                and username in getattr(f, "collaborators", [])
            )
        ]

class ConvoQueryService:
    """Read-access helpers for Conversation objects."""

    @staticmethod
    async def get_for_user(convo_id: str, user: dict) -> Optional[Conversation]:
        convo = await container.convo_repo.get_by_id(convo_id)
        if not convo:
            return None
        if AccessPolicy.can_read_convo(convo, user):
            return convo
        return None

    @staticmethod
    async def list_for_form(form_id: str, user: dict) -> List[Conversation]:
        all_convos = await container.convo_repo.get_by_form_id(form_id)
        if AccessPolicy.is_admin(user):
            return all_convos
        return [c for c in all_convos if getattr(c, "owner_id", None) in (None, user["user_id"])]

class ConversationParser:
    """Converts raw text lines into a structured history dict and back."""

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """
        Parse 'Speaker: message' lines into the internal history dict format.
        Delegates to the existing domain helper so the contract is unchanged.
        """
        return _parse_conversation_text(text)

    @staticmethod
    def render(history: Dict[str, Any]) -> str:
        """Render a history dict back to plain 'Speaker: message' text."""
        return render_history_for_model(history)

class SchemaBuilder:
    """Builds a FormSchema.fields dict from parallel name / question lists."""

    @staticmethod
    def from_pairs(
        field_names: List[str],
        field_values: List[str],
        *,
        autogenerate_question: bool = False,
    ) -> Dict[str, str]:
        return _build_schema_from_pairs(
            field_names, field_values, autogenerate_question=autogenerate_question
        )

    @staticmethod
    def default_question(field_name: str) -> str:
        return _default_field_question(field_name)

class FieldMerger:

    @staticmethod
    def merge_display(
        filled_data: Dict[str, Any] | None,
        accepted_new_fields: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return _merge_display_fields(filled_data, accepted_new_fields)

    @staticmethod
    def apply_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        _apply_field_overrides(target, overrides)

    @staticmethod
    def normalize_flat(payload: Dict[str, Any] | None) -> Dict[str, str]:
        return _normalize_flat_field_map(payload)

    @staticmethod
    def flatten_nested(source: Dict[str, Any]) -> Dict[str, str]:
        return _flatten_nested_field_map(source)

    @staticmethod
    def format_html(data: Dict[str, Any], prefix: str = "") -> str:
        return _format_filled_data(data, prefix)

class ExtractionService:

    @staticmethod
    async def extract(
        form: FormSchema,
        conversation_text: str,
        current_field_state: Dict[str, Any] | None = None,
        accepted_new_fields: Dict[str, Any] | None = None,
        replay_all_lines: bool = False,
    ) -> Dict[str, Any]:
        return await _extract_for_conversation_text(
            form,
            conversation_text,
            current_field_state,
            accepted_new_fields,
            replay_all_lines,
        )

class OutputRepository:
    """Persists and retrieves ExtractionResult records from MongoDB."""

    @staticmethod
    def collection():
        return container.convo_repo.db.outputs

    @staticmethod
    async def save(result: ExtractionResult, owner_id: str) -> None:
        doc = result.model_dump()
        doc["owner_id"] = owner_id
        await OutputRepository.collection().insert_one(doc)
        logger.info("Extraction result saved to Atlas.")

    @staticmethod
    async def find_by_run_id(run_id: str, user: dict) -> Optional[Dict[str, Any]]:
        query: Dict[str, Any] = {"run_id": run_id}
        if not AccessPolicy.is_admin(user):
            query["owner_id"] = user["user_id"]
        return await OutputRepository.collection().find_one(query)

    @staticmethod
    async def list_for_user(user: dict) -> List[Dict]:
        if AccessPolicy.is_admin(user):
            cursor = OutputRepository.collection().find({}).sort("_id", -1)
        else:
            cursor = OutputRepository.collection().find(
                {"owner_id": user["user_id"]}
            ).sort("_id", -1)
        return await cursor.to_list(length=100)

class TemplateRenderer:
    """Renders Jinja2 templates, always injecting current_user and is_admin."""

    @staticmethod
    def render(template_name: str, request: Request, ctx: dict, user=None):
        ctx.setdefault("request", request)
        ctx["current_user"] = user
        ctx["is_admin"]     = AccessPolicy.is_admin(user) if user else False
        return templates.TemplateResponse(request, template_name, ctx)

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

def _hash_password(password: str) -> str:
    return AuthService.hash_password(password)

def _verify_password(password: str, stored: str) -> bool:
    return AuthService.verify_password(password, stored)

def _make_session_token(user_id: str) -> str:
    return AuthService.make_session_token(user_id)

def _is_admin(user: dict) -> bool:
    return AccessPolicy.is_admin(user)

async def _get_current_user(request: Request) -> Optional[dict]:
    return await AuthService.get_current_user(request)

def _user_repo():
    return UserRepository.collection()

def _tmpl(template_name: str, request: Request, ctx: dict, user=None):
    return TemplateRenderer.render(template_name, request, ctx, user)

def _validate_username(username: str) -> Optional[str]:
    import asyncio
    if len(username) < 3 or len(username) > 32:
        return "Username must be 3-32 characters."
    if not re.match(r'^[a-zA-Z0-9_\-]+$', username):
        return "Username may only contain letters, numbers, underscores, and hyphens."
    return None

def _is_global(obj) -> bool:
    return AccessPolicy.is_global(obj)

def _can_write_form(form, user: dict) -> bool:
    return AccessPolicy.can_write_form(form, user)

def _can_write_convo(convo, user: dict) -> bool:
    return AccessPolicy.can_write_convo(convo, user)

async def _get_form_for_user(form_id: str, user: dict):
    return await FormQueryService.get_for_user(form_id, user)

async def _get_convo_for_user(convo_id: str, user: dict):
    return await ConvoQueryService.get_for_user(convo_id, user)


def _parse_conversation_text(text: str) -> Dict[str, str]:
    lines   = (text or "").strip().splitlines()
    history: Dict[str, str] = {}
    re_line = re.compile(r'^\s*([^:\n]{1,40})\s*:\s*(.*)\s*$')
    current_key = None
    turn_index = 0

    for line in lines:
        m = re_line.match(line)
        if m:
            speaker = m.group(1).strip()
            utterance = m.group(2).rstrip()
            turn_index += 1
            current_key = f"{speaker} {turn_index:06d}"
            history[current_key] = utterance
        elif current_key and line.strip():
            history[current_key] += "\n" + line.rstrip()

    return history


def _default_field_question(field_name: str) -> str:
    pretty = field_name.replace(".", " ").replace("_", " ").strip()
    pretty = " ".join(pretty.split())
    if not pretty:
        return "What value should be captured for this field?"
    starts_like_question = (
        "what", "when", "where", "who", "which", "why", "how",
        "is", "are", "do", "does", "did", "can", "could", "should",
        "would", "will", "has", "have", "had",
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
        name  = raw_name.strip()
        if not name:
            continue
        value = (raw_value or "").strip()
        if not value and autogenerate_question:
            value = _default_field_question(name)
        if not value:
            continue
        schema[name] = value
    return schema


def _set_nested_field(target: Dict[str, Any], field_key: str, value: Any) -> None:
    parts   = field_key.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _has_nested_field(source: Dict[str, Any], field_key: str) -> bool:
    parts   = field_key.split(".")
    current = source
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


def _flatten_nested_field_map(
    source: Dict[str, Any],
    prefix: str = "",
    out: Dict[str, str] | None = None,
) -> Dict[str, str]:
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
    replay_all_lines: bool = False,
) -> Dict[str, Any]:
    parsed    = _parse_conversation_text(conversation_text)
    full_convo = render_history_for_model(parsed)

    seeded_fields: Dict[str, Any] = {}
    for field_key in form.fields.keys():
        if current_field_state and field_key in current_field_state:
            candidate  = current_field_state[field_key]
            normalized = "" if candidate is None else str(candidate).strip()
            seeded_fields[field_key] = normalized if normalized else "N/A"
        else:
            seeded_fields[field_key] = "N/A"

    logger.info("[LiveExtract] Form=%s", form.name)
    logger.info("[LiveExtract] Form fields=%s", list(form.fields.keys()))
    logger.info("[LiveExtract] Current field state=%s", current_field_state or {})
    logger.info("[LiveExtract] Accepted new fields=%s", accepted_new_fields or {})
    logger.info("[LiveExtract] Seeded fields=%s", seeded_fields)
    logger.info("[LiveExtract] Parsed conversation=%s", parsed)

    model = container.pipeline.model

    if replay_all_lines:
        if not hasattr(model, "process_live_update"):
            raise RuntimeError(
                "Replay extraction requires a model with process_live_update."
            )
        field_keys    = list(form.fields.keys())
        lines         = [l for l in full_convo.strip().splitlines() if l.strip()]
        running_state = dict(seeded_fields)
        logger.info("[LiveExtract] Using iterative replay path with %s lines", len(lines))

        for index in range(len(lines)):
            answers = await model.process_live_update(
                conversation_text="\n".join(lines[: index + 1]),
                form_name=form.name,
                current_field_state=running_state,
                field_keys=field_keys,
                accepted_new_fields=_normalize_flat_field_map(accepted_new_fields),
            )
            running_state = dict(zip(field_keys, answers))
        answers_task = None

    elif hasattr(model, "process_live_update"):
        answers_task = model.process_live_update(
            conversation_text=full_convo.strip(),
            form_name=form.name,
            current_field_state=seeded_fields,
            field_keys=list(form.fields.keys()),
            accepted_new_fields=_normalize_flat_field_map(accepted_new_fields),
        )
        logger.info("[LiveExtract] Using process_live_update path")

    else:
        input_str = (
            f"Extract info from conversation to fill form.\n"
            f"Conversation: {full_convo}\n"
            f"Form: {form.name}\n"
            f"Fields: {json.dumps(seeded_fields)}"
        )
        answers_task = model.process_extraction_request(input_str)
        logger.info("[LiveExtract] Using full conversation replay path")

    summary_task = container.pipeline.summarizer.summarize(full_convo)

    if answers_task is None:
        summary = await summary_task
    else:
        answers, summary = await asyncio.gather(answers_task, summary_task)

    logger.info("[LiveExtract] Raw answers=%s", answers)
    logger.info("[LiveExtract] Summary=%s", summary)

    filled_data: Dict[str, Any]    = {}
    suggested_new_fields: Dict[str, str] = {}

    if isinstance(answers, dict):
        model_filled      = answers.get("filled_data", {})
        model_suggestions = answers.get("suggested_new_fields", {})
        flat_filled       = _flatten_nested_field_map(model_filled)       if isinstance(model_filled, dict)      else {}
        flat_suggestions  = _flatten_nested_field_map(model_suggestions)  if isinstance(model_suggestions, dict) else {}

        for field_key in form.fields.keys():
            value = flat_filled.get(field_key, seeded_fields.get(field_key, "N/A"))
            _set_nested_field(filled_data, field_key, value)

        for field_key, value in flat_suggestions.items():
            if field_key in form.fields:
                continue
            normalized_value = "" if value is None else str(value).strip()
            if normalized_value:
                suggested_new_fields[field_key] = normalized_value
    else:
        for field_key, value in zip(form.fields.keys(), answers):
            _set_nested_field(filled_data, field_key, value)

    for field_key in form.fields.keys():
        if not _has_nested_field(filled_data, field_key):
            _set_nested_field(filled_data, field_key, "N/A")

    normalized_accepted = _normalize_flat_field_map(accepted_new_fields)
    if normalized_accepted:
        filled_data = _merge_display_fields(filled_data, normalized_accepted)

    logger.info("[LiveExtract] Final filled_data=%s", filled_data)

    return {
        "filled_data":          filled_data,
        "suggested_new_fields": suggested_new_fields,
        "accepted_new_fields":  normalized_accepted,
        "summary":              summary,
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


async def _save_output(result: ExtractionResult, owner_id: str) -> None:
    await OutputRepository.save(result, owner_id)

async def _load_output_by_run_id(run_id: str, user: dict) -> Optional[Dict[str, Any]]:
    return await OutputRepository.find_by_run_id(run_id, user)

async def _load_outputs(user: dict) -> List[Dict]:
    return await OutputRepository.list_for_user(user)

async def _persist_conversation_and_extract(
    *,
    form_id: str,
    conversation_text: str,
    owner_id: str,
    conversation_id: str = "",
    conversation_name: str = "",
    reviewed_field_overrides: Dict[str, Any] | None = None,
    accepted_new_fields: Dict[str, Any] | None = None,
    version_metadata: Dict[str, Any] | None = None,
    use_live_extraction: bool = False,
) -> str:
    if not conversation_id.strip():
        conversation_id = str(uuid4())[:8]

    conversation_dict = ConversationParser.parse(conversation_text)
    if not conversation_dict:
        raise HTTPException(400, "Could not parse conversation.")

    version_payload: Dict[str, Any] = {
        "version_index": 0,
        "history": conversation_dict,
    }
    if version_metadata:
        version_payload.update(version_metadata)

    convo = Conversation(
        conversation_id=conversation_id,
        form_id=form_id,
        conversation_name=conversation_name.strip(),
        versions=[ConversationVersion(**version_payload)],
        owner_id=owner_id,
    )
    await container.convo_repo.save(convo)

    if use_live_extraction:
        form = await container.form_repo.get_by_id(form_id)
        if not form:
            raise HTTPException(404, "Form not found")
        extraction = await ExtractionService.extract(
            form,
            conversation_text,
            accepted_new_fields=accepted_new_fields,
            replay_all_lines=True,
        )
        result = ExtractionResult(
            conversation_id=conversation_id,
            form_id=form_id,
            filled_data=extraction.get("filled_data", {}),
            accepted_new_fields=accepted_new_fields or extraction.get("accepted_new_fields", {}),
            run_id=str(uuid4()),
            summary=str(extraction.get("summary", "")),
        )
        result.filled_data = FieldMerger.merge_display(result.filled_data, result.accepted_new_fields)
        if reviewed_field_overrides:
            FieldMerger.apply_overrides(result.filled_data, reviewed_field_overrides)
    else:
        result = await container.pipeline.run(
            conversation_id, form_id, version_index=0, owner_id=owner_id
        )
        if reviewed_field_overrides:
            FieldMerger.apply_overrides(result.filled_data, reviewed_field_overrides)
        if accepted_new_fields:
            result.accepted_new_fields = {
                key: "" if value is None else str(value)
                for key, value in accepted_new_fields.items()
                if isinstance(key, str) and key.strip()
            }
            result.filled_data = FieldMerger.merge_display(result.filled_data, result.accepted_new_fields)
        if reviewed_field_overrides or accepted_new_fields:
            await container.runlog_repo.update(result.run_id, {"extracted_fields": result.model_dump()})

    latest_version         = convo.versions[-1]
    latest_version.run_id  = result.run_id
    await container.convo_repo.save(convo)
    await OutputRepository.save(result, owner_id=owner_id)
    return conversation_id

class AuthHandler:

    @staticmethod
    async def register_page(request: Request):
        return templates.TemplateResponse(request, "register.html", {
            "request": request, "error": None, "prefill": {}
        })

    @staticmethod
    async def register(
        request: Request,
        email: str,
        username: str,
        password: str,
        confirm_password: str,
    ):
        prefill = {"email": email, "username": username}

        def _err(msg):
            return templates.TemplateResponse(request, "register.html", {
                "request": request, "error": msg, "prefill": prefill
            })

        if error := _validate_username(username):
            return _err(error)
        if username.lower() == ADMIN_USERNAME.lower():
            return _err("That username is reserved.")
        if len(password) < 8:
            return _err("Password must be at least 8 characters.")
        if password != confirm_password:
            return _err("Passwords do not match.")
        if await UserRepository.find_by_username(username):
            return _err("That username is already taken.")
        if await UserRepository.find_by_email(email):
            return _err("An account with that email already exists.")

        user_id = str(uuid4())
        await UserRepository.insert({
            "user_id":       user_id,
            "username":      username,
            "email":         email.lower(),
            "password_hash": AuthService.hash_password(password),
            "role":          "user",
            "created_at":    datetime.utcnow(),
        })
        logger.info(f"New user registered: {username} ({user_id})")

        token    = AuthService.make_session_token(user_id)
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax")
        return response

    @staticmethod
    async def login_page(request: Request, registered: str = ""):
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "error":   None,
            "success": "Account created! Please sign in." if registered == "1" else None,
            "prefill_username": "",
        })

    @staticmethod
    async def login(request: Request, username: str, password: str):
        user = await UserRepository.find_by_username(username)
        if not user or not AuthService.verify_password(password, user.get("password_hash", "")):
            return templates.TemplateResponse(request, "login.html", {
                "request": request,
                "error":   "Incorrect username or password.",
                "success": None,
                "prefill_username": username,
            })
        token    = AuthService.make_session_token(user["user_id"])
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax")
        logger.info(f"User logged in: {username} (role={user.get('role', 'user')})")
        return response

    @staticmethod
    async def logout():
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @staticmethod
    async def profile_page(request: Request):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        return TemplateRenderer.render("profile.html", request, {
            "username_error": None, "username_success": None,
            "password_error": None, "password_success": None,
        }, user)

    @staticmethod
    async def change_username(request: Request, new_username: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        new_username = new_username.strip()
        ctx = {"username_error": None, "username_success": None,
               "password_error": None, "password_success": None}
        if error := _validate_username(new_username):
            ctx["username_error"] = error
            return TemplateRenderer.render("profile.html", request, ctx, user)
        if new_username == user["username"]:
            ctx["username_error"] = "That's already your current username."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        if new_username.lower() == ADMIN_USERNAME.lower() and not AccessPolicy.is_admin(user):
            ctx["username_error"] = "That username is reserved."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        if await UserRepository.find_by_username(new_username):
            ctx["username_error"] = "That username is already taken."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        await UserRepository.update({"user_id": user["user_id"]}, {"$set": {"username": new_username}})
        user["username"]          = new_username
        ctx["username_success"]   = "Username updated successfully."
        return TemplateRenderer.render("profile.html", request, ctx, user)

    @staticmethod
    async def change_password(
        request: Request,
        old_password: str,
        new_password: str,
        confirm_new_password: str,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        ctx = {"username_error": None, "username_success": None,
               "password_error": None, "password_success": None}
        if not AuthService.verify_password(old_password, user.get("password_hash", "")):
            ctx["password_error"] = "Current password is incorrect."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        if len(new_password) < 8:
            ctx["password_error"] = "New password must be at least 8 characters."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        if new_password != confirm_new_password:
            ctx["password_error"] = "New passwords do not match."
            return TemplateRenderer.render("profile.html", request, ctx, user)
        await UserRepository.update(
            {"user_id": user["user_id"]},
            {"$set": {"password_hash": AuthService.hash_password(new_password)}},
        )
        ctx["password_success"] = "Password updated successfully."
        return TemplateRenderer.render("profile.html", request, ctx, user)

class AdminHandler:
    """Admin-only user management."""

    @staticmethod
    async def list_users(request: Request):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        if not AccessPolicy.is_admin(user):
            raise HTTPException(403, "Admin access required.")
        all_users = await UserRepository.list_all()
        return TemplateRenderer.render("admin_users.html", request, {"all_users": all_users}, user)

    @staticmethod
    async def set_role(request: Request, target_user_id: str, role: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        if not AccessPolicy.is_admin(user):
            raise HTTPException(403, "Admin access required.")
        if role not in ("admin", "user"):
            raise HTTPException(400, "Invalid role.")
        if target_user_id == user["user_id"] and role != "admin":
            raise HTTPException(400, "You cannot remove your own admin role.")
        await UserRepository.update({"user_id": target_user_id}, {"$set": {"role": role}})
        logger.info(f"Admin {user['username']} set role={role} for user_id={target_user_id}")
        return RedirectResponse(url="/admin/users", status_code=303)

    @staticmethod
    async def delete_user(request: Request, target_user_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        if not AccessPolicy.is_admin(user):
            raise HTTPException(403, "Admin access required.")
        if target_user_id == user["user_id"]:
            raise HTTPException(400, "You cannot delete your own account.")
        await UserRepository.delete(target_user_id)
        logger.info(f"Admin {user['username']} deleted user_id={target_user_id}")
        return RedirectResponse(url="/admin/users", status_code=303)

class FormHandler:

    @staticmethod
    async def home(request: Request):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        visible_forms      = await FormQueryService.get_all_visible(user)
        deletable_form_ids = {f.id for f in visible_forms if getattr(f, "owner_id", None) == user["user_id"]}
        form_owners        = {}
        if AccessPolicy.is_admin(user):
            all_user_ids = {f.owner_id for f in visible_forms if f.owner_id}
            if all_user_ids:
                docs       = await UserRepository.find_by_ids(list(all_user_ids))
                form_owners = {d["user_id"]: d["username"] for d in docs}
        return TemplateRenderer.render("home.html", request, {
            "forms":              visible_forms,
            "deletable_form_ids": deletable_form_ids,
            "form_owners":        form_owners,
        }, user)

    @staticmethod
    async def new_form_page(request: Request):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        return TemplateRenderer.render("create_form.html", request, {"is_admin": AccessPolicy.is_admin(user)}, user)

    @staticmethod
    async def create_form(
        request: Request,
        form_name: str,
        form_description: str,
        visibility: str,
        field_names: List[str],
        field_types: List[str],
        collaborators: List[str],
    ):
        """
        Sequence:
          → AuthService.get_current_user
          → SchemaBuilder.from_pairs
          → container.form_repo.save
          → RedirectResponse /forms/{form_id}
        """
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)

        schema_dict = SchemaBuilder.from_pairs(field_names, field_types, autogenerate_question=False)
        if not schema_dict:
            raise HTTPException(400, "At least one valid field is required.")

        form_id    = str(uuid4())[:8]
        owner_id   = user["user_id"]
        clean_name = form_name.strip()
        if not clean_name:
            raise HTTPException(400, "Form name is required.")

        clean_collabs = [c.strip() for c in (collaborators or []) if c.strip()]

        await container.form_repo.save(FormSchema(
            form_id=form_id,
            form_name=clean_name,
            description=form_description.strip(),
            schema=schema_dict,
            owner_id=owner_id,
            visibility=visibility,
            collaborators=clean_collabs if visibility == "collaborative" else [],
        ))
        logger.info(f"Form created: {form_id} by {user['username']}")
        return RedirectResponse(url=f"/forms/{form_id}", status_code=303)

    @staticmethod
    async def view_form(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        return TemplateRenderer.render("view_form.html", request, {"form": form}, user)

    @staticmethod
    async def edit_form_page(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raise HTTPException(404, "Form not found")
        can_save_in_place = AccessPolicy.can_write_form(form, user)
        return TemplateRenderer.render("edit_form.html", request, {
            "form": form, "can_save_in_place": can_save_in_place
        }, user)

    @staticmethod
    async def save_form_edits(
        request: Request,
        form_id: str,
        form_name: str,
        form_description: str,
        field_names: List[str],
        field_instructions: List[str],
        save_mode: str,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        existing = await FormQueryService.get_for_user(form_id, user)
        if not existing:
            raise HTTPException(404, "Form not found")
        if save_mode == "save" and not AccessPolicy.can_write_form(existing, user):
            raise HTTPException(403, "You can only save a copy of this form using 'Save As New Form'.")

        schema_dict = SchemaBuilder.from_pairs(field_names, field_instructions, autogenerate_question=True)
        if not schema_dict:
            raise HTTPException(400, "At least one valid field is required.")

        requested_name      = form_name.strip()
        cleaned_description = form_description.strip()

        if save_mode == "save":
            target_form_id = form_id
            cleaned_name   = requested_name or existing.name
            owner          = getattr(existing, "owner_id", user["user_id"])
        elif save_mode == "save_as":
            cleaned_name = requested_name
            if not cleaned_name:
                raise HTTPException(400, "Please provide a new form name when using Save As.")
            if cleaned_name.lower() == existing.name.strip().lower():
                raise HTTPException(400, "Please change the form name when using Save As.")
            target_form_id = str(uuid4())[:8]
            owner          = user["user_id"]
        else:
            raise HTTPException(400, "Invalid save mode.")

        await container.form_repo.save(FormSchema(
            form_id=target_form_id,
            form_name=cleaned_name,
            description=cleaned_description,
            schema=schema_dict,
            owner_id=owner,
        ))
        return RedirectResponse(url=f"/forms/{target_form_id}", status_code=303)

    @staticmethod
    async def delete_form(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await container.form_repo.get_by_id(form_id)
        if not form:
            raise HTTPException(404, "Form not found")
        if not AccessPolicy.can_write_form(form, user):
            raise HTTPException(403, "You don't have permission to delete this form.")
        await container.form_repo.delete_by_id(form_id)
        logger.info(f"Form deleted: {form_id} by {user['username']}")
        return RedirectResponse(url="/", status_code=303)

    @staticmethod
    async def delete_form_api(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        form = await container.form_repo.get_by_id(form_id)
        if not form:
            raise HTTPException(404, "Form not found")
        if not AccessPolicy.can_write_form(form, user):
            raise HTTPException(403, "You don't have permission to delete this form.")
        await container.form_repo.delete_by_id(form_id)
        return JSONResponse({"ok": True})

class ConversationHandler:

    @staticmethod
    async def enter_conversation_page(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        return TemplateRenderer.render("enter_conversation.html", request, {"form": form}, user)

    @staticmethod
    async def create_conversation(
        request: Request,
        form_id: str,
        conversation_id: str,
        conversation_name: str,
        field_overrides_json: str,
        accepted_new_fields_json: str,
        conversation_text: str,
        extract: bool,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)

        if not conversation_id.strip():
            conversation_id = str(uuid4())[:8]

        reviewed_field_overrides: Dict[str, Any] = {}
        if field_overrides_json.strip():
            try:
                parsed_overrides = json.loads(field_overrides_json)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid reviewed fields payload: {str(e)}")
            if not isinstance(parsed_overrides, dict):
                raise HTTPException(400, "Reviewed fields payload must be a JSON object.")
            reviewed_field_overrides = parsed_overrides

        accepted_new_fields: Dict[str, Any] = {}
        if accepted_new_fields_json.strip():
            try:
                parsed_new_fields = json.loads(accepted_new_fields_json)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid accepted new fields payload: {str(e)}")
            if not isinstance(parsed_new_fields, dict):
                raise HTTPException(400, "Accepted new fields payload must be a JSON object.")
            accepted_new_fields = {
                key: "" if value is None else str(value)
                for key, value in parsed_new_fields.items()
                if isinstance(key, str) and key.strip()
            }

        if extract:
            conversation_id = await _persist_conversation_and_extract(
                form_id=form_id,
                conversation_text=conversation_text,
                owner_id=user["user_id"],
                conversation_id=conversation_id,
                conversation_name=conversation_name,
                reviewed_field_overrides=reviewed_field_overrides,
                accepted_new_fields=accepted_new_fields,
                version_metadata={"source_mode": "text"},
            )
            return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)
        else:
            conversation_dict = ConversationParser.parse(conversation_text)
            if not conversation_dict:
                raise HTTPException(400, "Could not parse conversation.")
            convo = Conversation(
                conversation_id=conversation_id,
                form_id=form_id,
                conversation_name=conversation_name.strip(),
                versions=[ConversationVersion(version_index=0, history=conversation_dict, source_mode="text")],
                owner_id=user["user_id"],
            )
            await container.convo_repo.save(convo)
            return RedirectResponse(url=f"/conversations/{conversation_id}?form_id={form_id}", status_code=303)

    @staticmethod
    async def list_conversations(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        convos = await ConvoQueryService.list_for_form(form_id, user)
        return TemplateRenderer.render("list_conversations.html", request, {"form": form, "convos": convos}, user)

    @staticmethod
    async def view_conversation(request: Request, convo_id: str, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        convo = await ConvoQueryService.get_for_user(convo_id, user)
        if not convo:
            raise HTTPException(404, "Conversation not found")
        return TemplateRenderer.render("view_conversation.html", request, {
            "convo": convo,
            "form_id": form_id,
            "back_link": f"/forms/{form_id}/conversations",
        }, user)

    @staticmethod
    async def edit_conversation_page(request: Request, convo_id: str, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        convo = await ConvoQueryService.get_for_user(convo_id, user)
        form  = await FormQueryService.get_for_user(form_id, user)
        if not convo:
            raise HTTPException(404, "Conversation not found")
        if not form:
            raise HTTPException(404, "Form not found")
        raw_text = ConversationParser.render(convo.latest_history)
        return TemplateRenderer.render("edit_conversation.html", request, {
            "convo": convo, "raw_text": raw_text, "form_id": form_id, "form": form,
        }, user)

    @staticmethod
    async def update_conversation(
        request: Request,
        convo_id: str,
        form_id: str,
        new_content: str,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        existing = await ConvoQueryService.get_for_user(convo_id, user)
        if not existing:
            raise HTTPException(404, "Conversation not found")
        new_history = ConversationParser.parse(new_content)
        if not new_history:
            raise HTTPException(400, "Could not parse updated conversation.")
        new_v = ConversationVersion(version_index=len(existing.versions), history=new_history)
        existing.versions.append(new_v)
        await container.convo_repo.save(existing)
        return RedirectResponse(url=f"/extract/{form_id}/{convo_id}", status_code=303)

    @staticmethod
    async def enter_collab_conversation(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        room_id = f"form-{form_id}"
        return TemplateRenderer.render("enter_conversation_collab.html", request, {
            "form": form, "room_id": room_id, "current_user": user["username"],
        }, user)

class ExtractionHandler:

    @staticmethod
    async def run_extraction(request: Request, form_id: str, convo_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        try:
            convo = await ConvoQueryService.get_for_user(convo_id, user)
            if not convo or not convo.versions:
                raise HTTPException(404, "No conversation history found.")
            form = await FormQueryService.get_for_user(form_id, user)
            if not form:
                raise HTTPException(404, "Form not found")

            latest_version = convo.versions[-1]
            result = await container.pipeline.run(
                convo_id, form_id,
                version_index=latest_version.version_index,
                owner_id=user["user_id"],
            )

            latest_version.run_id = result.run_id
            await container.convo_repo.save(convo)
            await OutputRepository.save(result, owner_id=user["user_id"])

            display_fields = FieldMerger.merge_display(result.filled_data, result.accepted_new_fields)
            result_payload = {
                "filled_data":          display_fields,
                "accepted_new_fields":  result.accepted_new_fields,
            }
            return TemplateRenderer.render("run_extraction.html", request, {
                "form_id":       form_id,
                "convo_id":      convo_id,
                "convo":         convo,
                "fields_html":   FieldMerger.format_html(display_fields),
                "new_fields_html": FieldMerger.format_html(result.accepted_new_fields),
                "json_pretty":   json.dumps(result_payload, indent=2),
                "result":        result,
                "summary":       result.summary,
            }, user)
        except Exception as e:
            raise HTTPException(500, str(e))

    @staticmethod
    async def preview_extraction(
        request: Request,
        form_id: str,
        conversation_text: str,
        field_state_json: str,
        accepted_new_fields_json: str,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raise HTTPException(404, "Form not found")
        if not conversation_text.strip():
            return JSONResponse({"filled_data": {}, "summary": ""})

        current_field_state: Dict[str, Any] = {}
        accepted_new_fields: Dict[str, Any] = {}

        if field_state_json.strip():
            try:
                parsed_state = json.loads(field_state_json)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid field state payload: {str(e)}")
            if not isinstance(parsed_state, dict):
                raise HTTPException(400, "Field state payload must be a JSON object.")
            allowed_keys = set(form.fields.keys())
            for key, value in parsed_state.items():
                if isinstance(key, str) and key in allowed_keys:
                    current_field_state[key] = "" if value is None else str(value)

        if accepted_new_fields_json.strip():
            try:
                parsed_new_fields = json.loads(accepted_new_fields_json)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"Invalid accepted new fields payload: {str(e)}")
            if not isinstance(parsed_new_fields, dict):
                raise HTTPException(400, "Accepted new fields payload must be a JSON object.")
            for key, value in parsed_new_fields.items():
                if isinstance(key, str) and key.strip():
                    accepted_new_fields[key.strip()] = "" if value is None else str(value)

        logger.info("[PreviewExtract] form_id=%s", form_id)
        try:
            extraction = await ExtractionService.extract(
                form,
                conversation_text,
                current_field_state,
                accepted_new_fields,
            )
            return JSONResponse(extraction)
        except Exception as e:
            logger.exception("[PreviewExtract] failed")
            raise HTTPException(500, str(e))

    @staticmethod
    async def api_live_extract(request: Request, payload: LiveExtractRequest):
        user = await AuthService.get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        form = await FormQueryService.get_for_user(payload.form_id, user)
        if not form:
            raise HTTPException(404, "Form not found")
        context = payload.conversation.strip()
        if not context:
            raise HTTPException(400, "Conversation text is empty")
        extraction = await ExtractionService.extract(form, context)
        return JSONResponse(content=extraction.get("filled_data", {}))

    @staticmethod
    async def live_extraction_page(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        return TemplateRenderer.render("static_enter_conversation.html", request, {"form": form}, user)

class ASRHandler:

    @staticmethod
    async def asr_extraction_page(request: Request, form_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        form = await FormQueryService.get_for_user(form_id, user)
        if not form:
            raw_form = await container.form_repo.get_by_id(form_id)
            if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                    and user["username"] in getattr(raw_form, "collaborators", []):
                form = raw_form
        if not form:
            raise HTTPException(404, "Form not found")
        return TemplateRenderer.render("asr_enter_conversation.html", request, {
            "form": form, "input_languages": SUPPORTED_INPUT_LANGUAGES,
        }, user)

    @staticmethod
    async def create_conversation_asr(
        request: Request,
        form_id: str = Form(...),
        input_language: str = Form("en"),
        conversation_id: str = Form(""),
        conversation_name: str = Form(""),
        conversation_text: str = Form(""),
        translated_text_override: str = Form(""),
        raw_transcript_override: str = Form(""),
        num_speakers: int = Form(0),
        audio_file: UploadFile | None = File(default=None),
    ):
        user = await _get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)

        form = await _get_form_for_user(form_id, user)
        if not form:
            raise HTTPException(404, "Form not found")

        transcript_text = ""
        translated_text = ""
        diarized_turns: list = []

        if translated_text_override.strip():
            translated_text = translated_text_override.strip()
            transcript_text = raw_transcript_override.strip() or translated_text
        elif audio_file is not None and (audio_file.filename or "").strip():
            raw_audio = await audio_file.read()
            if not raw_audio:
                raise HTTPException(400, "Uploaded audio file is empty.")

            if num_speakers and num_speakers >= 2:
                # Diarization path: identify who spoke when
                diarized_turns = await container.diarizer.diarize(
                    audio_bytes=raw_audio,
                    filename=audio_file.filename,
                    num_speakers=num_speakers,
                    input_language=input_language,
                )
                raw_joined = " ".join(t["text"] for t in diarized_turns)
                transcript_text = raw_joined
                translated_text = _diarized_to_conversation_text(diarized_turns)
            else:
                # Standard single-speaker path
                transcript_text = await container.asr_transcriber.transcribe_to_text(
                    audio_bytes=raw_audio,
                    filename=audio_file.filename,
                    input_language=input_language,
                )
                translated_text = await container.translator.translate_to_english(transcript_text, input_language)
        elif conversation_text.strip():
            transcript_text = conversation_text
            translated_text = await container.translator.translate_to_english(transcript_text, input_language)
        else:
            raise HTTPException(400, "Please upload an audio file or record audio.")

        # Use diarized conversation text if available, otherwise fall back to
        # the single-speaker flat transcript conversion.
        if diarized_turns:
            conversation_payload = translated_text  # already formatted with speaker labels
        else:
            conversation_payload = _transcript_to_conversation_text(translated_text)

        final_text = (translated_text_override or "").strip() or (raw_transcript_override or "").strip()
        if final_text and num_speakers == 0 and "Speaker" not in final_text[:20]:
            final_text = f"Speaker: {final_text}"
            conversation_payload = final_text

        should_use_live = num_speakers > 0

        if not conversation_payload:
            raise HTTPException(400, "Transcription produced empty text.")

        try:
            conversation_id = await _persist_conversation_and_extract(
                form_id=form_id,
                conversation_text=conversation_payload,
                owner_id=user["user_id"],
                conversation_id=conversation_id,
                conversation_name=conversation_name,
                reviewed_field_overrides=None,
                version_metadata={
                    "source_mode": "asr",
                    "input_language": input_language,
                    "raw_transcript": transcript_text,
                    "translated_transcript": translated_text,
                },
                use_live_extraction=should_use_live,
            )
        except Exception as e:
            raise HTTPException(500, f"Conversation saved, but extraction failed: {str(e)}")

        return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)

    @staticmethod
    async def transcribe_live_audio(
        request: Request,
        input_language: str,
        audio_file: UploadFile | None,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        if audio_file is None or not (audio_file.filename or "").strip():
            raise HTTPException(400, "Audio file is required.")
        raw_audio = await audio_file.read()
        if not raw_audio:
            raise HTTPException(400, "Uploaded audio file is empty.")

        try:
            transcript_text = await container.stt_service.transcribe_to_text(
                audio_bytes=raw_audio,
                filename=audio_file.filename,
                input_language=input_language,
            )
        except RuntimeError as exc:
            message = str(exc)
            logger.warning("[STT] Runtime issue: %s", message)
            if "Whisper STT is temporarily disabled" in message:
                raise HTTPException(503, message) from exc
            recoverable = ("audio preprocessing failed", "audio decode failed",
                           "speechrecognition decode failed", "unknown format",
                           "cannot read", "file does not start")
            if any(m in message.lower() for m in recoverable):
                return JSONResponse({"text": "", "raw_text": "", "warning": message})
            raise HTTPException(500, message) from exc
        except Exception as exc:
            logger.exception("[STT] Unexpected transcription failure")
            raise HTTPException(500, f"STT failed: {str(exc)}") from exc

        try:
            translated_text = await container.translator.translate_to_english(transcript_text, input_language)
        except Exception as exc:
            logger.exception("[STT] Translation step failed")
            raise HTTPException(500, f"STT translation failed: {str(exc)}") from exc

        return JSONResponse({"text": translated_text.strip(), "raw_text": transcript_text.strip()})

    @staticmethod
    async def asr_translate_preview(
        request: Request,
        input_language: str,
        audio_file: UploadFile | None,
    ):
        user = await AuthService.get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        if audio_file is None or not (audio_file.filename or "").strip():
            raise HTTPException(400, "Audio file is required.")
        raw_audio = await audio_file.read()
        if not raw_audio:
            raise HTTPException(400, "Uploaded audio file is empty.")

        transcript_text = await container.asr_transcriber.transcribe_to_text(
            audio_bytes=raw_audio,
            filename=audio_file.filename,
            input_language=input_language,
        )
        translated_text = await container.translator.translate_to_english(transcript_text, input_language)
        return JSONResponse({"raw_text": transcript_text.strip(), "translated_text": translated_text.strip()})

class OutputHandler:
    """View saved extraction outputs and pipeline run logs."""

    @staticmethod
    async def view_outputs(request: Request):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        return TemplateRenderer.render("view_outputs.html", request, {
            "outputs": await OutputRepository.list_for_user(user)
        }, user)

    @staticmethod
    async def view_output_detail(request: Request, run_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)

        out          = await OutputRepository.find_by_run_id(run_id, user)
        version_index = None

        if not out:
            run_log = await container.runlog_repo.get_by_id(run_id)
            if not run_log:
                raise HTTPException(404, "Output not found")
            ef = getattr(run_log, "extracted_fields", {}) or {}
            if not AccessPolicy.is_admin(user) and getattr(run_log, "owner_id", None) not in (None, user["user_id"]):
                raise HTTPException(404, "Output not found")
            version_index = getattr(run_log, "version_index", None)
            out = {
                "run_id":              run_log.run_id,
                "conversation_id":     run_log.conversation_id,
                "form_id":             ef.get("form_id", ""),
                "filled_data":         FieldMerger.merge_display(ef.get("filled_data", {}), ef.get("accepted_new_fields", {})),
                "accepted_new_fields": ef.get("accepted_new_fields", {}),
                "summary":             getattr(run_log, "summary", "") or ef.get("summary", ""),
            }

        history          = {}
        raw_conversation = ""
        convo_missing    = False
        convo_id         = out.get("conversation_id", "")
        if convo_id:
            convo = await ConvoQueryService.get_for_user(convo_id, user)
            if convo:
                if version_index is not None:
                    matched = next(
                        (v for v in convo.versions if v.version_index == version_index), None
                    )
                    history = (matched.history if matched else convo.latest_history) or {}
                else:
                    history = convo.latest_history or {}
            else:
                raw_conversation = out.get("raw_conversation", "")
                convo_missing    = True

        return TemplateRenderer.render("view_output_detail.html", request, {
            "out": out, "history": history,
            "raw_conversation": raw_conversation,
            "convo_missing": convo_missing,
            "version_index": version_index,
        }, user)

    @staticmethod
    async def view_runs(request: Request, limit: int = 50):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        if not AccessPolicy.is_admin(user):
            raise HTTPException(403, "Run logs are only available to administrators.")
        runs = await container.runlog_repo.get_recent(limit)
        return TemplateRenderer.render("view_runs.html", request, {"runs": runs}, user)

    @staticmethod
    async def view_run(request: Request, run_id: str):
        user = await AuthService.get_current_user(request)
        if not user:
            return RedirectResponse(url="/login", status_code=303)
        if not AccessPolicy.is_admin(user):
            raise HTTPException(403, "Run logs are only available to administrators.")
        r = await container.runlog_repo.get_by_id(run_id)
        if not r:
            raise HTTPException(404, "Run not found")
        ef             = getattr(r, "extracted_fields", {}) or {}
        display_fields = FieldMerger.merge_display(ef.get("filled_data", {}), ef.get("accepted_new_fields", {}))
        return TemplateRenderer.render("view_run.html", request, {
            "run": r, "ef": ef,
            "fields_html": FieldMerger.format_html(display_fields),
        }, user)
    
def _diarized_to_conversation_text(diarized_turns: list) -> str:
    """
    Convert a list of {"speaker": "SPEAKER 1", "text": "..."} dicts produced by
    LocalSpeakerDiarizer into the pipe-delimited conversation text format used
    by the rest of the system (same format as _parse_conversation_text expects).
    """
    lines = []
    for turn in diarized_turns:
        speaker = (turn.get("speaker") or "Speaker").strip()
        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)
