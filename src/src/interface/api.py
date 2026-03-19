"""FastAPI Interface - Routes only, DI handled by dependencies.py"""
import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from ..domain.domain import Conversation, FormSchema, ExtractionResult, ConversationVersion
from .dependencies import container, Container

from pydantic import BaseModel as PydanticBaseModel
from ..infrastructure.ai.local_model import LocalHuggingFaceModel
from ..domain.domain import ExtractionRequest

# ── Auth imports ──────────────────────────────────────────────────────────────
import hashlib
import secrets
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

_live_model: LocalHuggingFaceModel | None = None

def get_live_model() -> LocalHuggingFaceModel:
    """Return (and lazily initialise) the module-level LocalHuggingFaceModel."""
    global _live_model
    if _live_model is None:
        _live_model = LocalHuggingFaceModel()
    return _live_model

class LiveExtractRequest(PydanticBaseModel):
    form_id: str
    conversation: str   # raw "Speaker: text\nSpeaker: text\n…" string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="src/interface/templates")

# ── Session / auth helpers ────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_signer = URLSafeTimedSerializer(SECRET_KEY)
SESSION_COOKIE = "pl_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 14  # 14 days


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


async def _get_current_user(request: Request) -> Optional[dict]:
    """Return the current user dict (from MongoDB) or None."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = _read_session_token(token)
    if not user_id:
        return None
    return await _user_repo().find_one({"user_id": user_id})


async def _require_user(request: Request) -> dict:
    """Dependency: redirect to /login if not authenticated."""
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user


def _user_repo():
    """Return the MongoDB 'users' collection via the container's db."""
    return container.convo_repo.db.users


def _tmpl(template_name: str, request: Request, ctx: dict, user=None):
    """Shorthand for TemplateResponse that always injects current_user."""
    ctx.setdefault("request", request)
    ctx["current_user"] = user
    return templates.TemplateResponse(template_name, ctx)


# ── Seed / lifespan ───────────────────────────────────────────────────────────

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
    autogenerate_question: bool = False
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
        if '$date' in obj:
            s = obj['$date']
            if isinstance(s, dict) and '$numberLong' in s:
                s = s['$numberLong']
            if isinstance(s, str):
                if s.endswith('Z'):
                    s = s.replace('Z', '+00:00')
                try:
                    return datetime.fromisoformat(s)
                except Exception:
                    return s
        if '$oid' in obj:
            return str(obj['$oid'])
        return {k: _convert_mongo_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_mongo_types(i) for i in obj]
    else:
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
                    data["versions"] = [ConversationVersion(version_index=0, history=history).model_dump()]
                # Mark as global starter (no owner)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    Container.initialize()
    await container.runlog_repo.ensure_indexes()
    # Ensure a unique index on username and email for the users collection
    await _user_repo().create_index("username", unique=True)
    await _user_repo().create_index("email", unique=True)
    await seed_data()
    logger.info("Application started.")
    yield


app = FastAPI(title="ProductLabs Form Filler", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/interface/static"), name="static")


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None, "prefill": {}})


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    prefill = {"email": email, "username": username}

    # Validation
    if len(username) < 3 or len(username) > 32:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username must be 3–32 characters.",
            "prefill": prefill
        })
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', username):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username may only contain letters, numbers, underscores, and hyphens.",
            "prefill": prefill
        })
    if len(password) < 8:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Password must be at least 8 characters.",
            "prefill": prefill
        })
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Passwords do not match.",
            "prefill": prefill
        })

    # Check uniqueness
    existing_username = await _user_repo().find_one({"username": username})
    if existing_username:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "That username is already taken.",
            "prefill": prefill
        })
    existing_email = await _user_repo().find_one({"email": email.lower()})
    if existing_email:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "An account with that email already exists.",
            "prefill": prefill
        })

    user_id = str(uuid4())
    await _user_repo().insert_one({
        "user_id": user_id,
        "username": username,
        "email": email.lower(),
        "password_hash": _hash_password(password),
        "created_at": datetime.utcnow()
    })
    logger.info(f"New user registered: {username} ({user_id})")

    # Auto-login after registration
    token = _make_session_token(user_id)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_MAX_AGE, httponly=True, samesite="lax"
    )
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, registered: str = ""):
    success_msg = "Account created! Please sign in." if registered == "1" else None
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None,
        "success": success_msg,
        "prefill_username": ""
    })


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    user = await _user_repo().find_one({"username": username})
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Incorrect username or password.",
            "success": None,
            "prefill_username": username
        })

    token = _make_session_token(user["user_id"])
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_MAX_AGE, httponly=True, samesite="lax"
    )
    logger.info(f"User logged in: {username}")
    return response


@app.post("/logout", response_class=RedirectResponse)
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ── Profile routes ────────────────────────────────────────────────────────────

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return _tmpl("profile.html", request, {
        "username_error": None, "username_success": None,
        "password_error": None, "password_success": None
    }, user=user)


@app.post("/profile/change-username", response_class=HTMLResponse)
async def change_username(
    request: Request,
    new_username: str = Form(...)
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    import re
    new_username = new_username.strip()
    ctx = {
        "username_error": None, "username_success": None,
        "password_error": None, "password_success": None
    }

    if len(new_username) < 3 or len(new_username) > 32:
        ctx["username_error"] = "Username must be 3–32 characters."
        return _tmpl("profile.html", request, ctx, user=user)

    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_username):
        ctx["username_error"] = "Username may only contain letters, numbers, underscores, and hyphens."
        return _tmpl("profile.html", request, ctx, user=user)

    if new_username == user["username"]:
        ctx["username_error"] = "That's already your current username."
        return _tmpl("profile.html", request, ctx, user=user)

    existing = await _user_repo().find_one({"username": new_username})
    if existing:
        ctx["username_error"] = "That username is already taken."
        return _tmpl("profile.html", request, ctx, user=user)

    await _user_repo().update_one(
        {"user_id": user["user_id"]},
        {"$set": {"username": new_username}}
    )
    user["username"] = new_username
    ctx["username_success"] = "Username updated successfully."
    return _tmpl("profile.html", request, ctx, user=user)


@app.post("/profile/change-password", response_class=HTMLResponse)
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_new_password: str = Form(...)
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    ctx = {
        "username_error": None, "username_success": None,
        "password_error": None, "password_success": None
    }

    if not _verify_password(old_password, user.get("password_hash", "")):
        ctx["password_error"] = "Current password is incorrect."
        return _tmpl("profile.html", request, ctx, user=user)

    if len(new_password) < 8:
        ctx["password_error"] = "New password must be at least 8 characters."
        return _tmpl("profile.html", request, ctx, user=user)

    if new_password != confirm_new_password:
        ctx["password_error"] = "New passwords do not match."
        return _tmpl("profile.html", request, ctx, user=user)

    await _user_repo().update_one(
        {"user_id": user["user_id"]},
        {"$set": {"password_hash": _hash_password(new_password)}}
    )
    ctx["password_success"] = "Password updated successfully."
    return _tmpl("profile.html", request, ctx, user=user)


# ── Main routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - list all forms owned by the current user + global starters."""
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    all_forms = await container.form_repo.get_all()
    # Show global starter forms (owner_id is None) + user's own forms
    forms = [
        f for f in all_forms
        if getattr(f, "owner_id", None) in (None, user["user_id"])
    ]
    return _tmpl("home.html", request, {"forms": forms}, user=user)


@app.post("/forms", response_class=RedirectResponse)
async def handle_create_form(
    request: Request,
    form_name: str = Form(...),
    form_description: str = Form(""),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_type: List[str] = Form(..., alias="field_type[]")
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    form_id = str(uuid4())[:8]
    schema_dict = {name: ftype for name, ftype in zip(field_name, field_type) if name.strip()}
    new_form = FormSchema(
        form_id=form_id,
        form_name=form_name,
        description=form_description,
        schema=schema_dict,
        owner_id=user["user_id"]
    )
    await container.form_repo.save(new_form)
    logger.info(f"Form {form_id} created by {user['username']}.")
    return RedirectResponse(url="/", status_code=303)


@app.get("/forms/new", response_class=HTMLResponse)
async def create_form(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return _tmpl("create_form.html", request, {}, user=user)


@app.get("/forms/{form_id}", response_class=HTMLResponse)
async def view_form(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("view_form.html", request, {"form": form}, user=user)


@app.get("/forms/{form_id}/edit", response_class=HTMLResponse)
async def edit_form(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("edit_form.html", request, {"form": form}, user=user)


@app.post("/forms/{form_id}/edit", response_class=RedirectResponse)
async def save_form_edits(
    request: Request,
    form_id: str,
    form_name: str = Form(...),
    form_description: str = Form(""),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_instruction: List[str] = Form(..., alias="field_instruction[]"),
    save_mode: str = Form("save")
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    existing = await _get_form_for_user(form_id, user)
    if not existing:
        raise HTTPException(404, "Form not found")

    schema_dict = _build_schema_from_pairs(field_name, field_instruction, autogenerate_question=True)
    if not schema_dict:
        raise HTTPException(400, "At least one valid field is required.")

    requested_name = form_name.strip()
    cleaned_description = form_description.strip()

    if save_mode == "save":
        target_form_id = form_id
        cleaned_name = requested_name or existing.name
        owner = getattr(existing, "owner_id", user["user_id"])
    elif save_mode == "save_as":
        cleaned_name = requested_name
        if not cleaned_name:
            raise HTTPException(400, "Please provide a new form name when using Save As.")
        if cleaned_name.lower() == existing.name.strip().lower():
            raise HTTPException(400, "Please change the form name when using Save As.")
        target_form_id = str(uuid4())[:8]
        owner = user["user_id"]
    else:
        raise HTTPException(400, "Invalid save mode.")

    updated_form = FormSchema(
        form_id=target_form_id,
        form_name=cleaned_name,
        description=cleaned_description,
        schema=schema_dict,
        owner_id=owner
    )
    await container.form_repo.save(updated_form)
    return RedirectResponse(url=f"/forms/{target_form_id}", status_code=303)


@app.post("/forms/{form_id}/delete", response_class=RedirectResponse)
async def delete_form(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    # Only the owner (or global forms with no owner) can be deleted by any user
    form_owner = getattr(form, "owner_id", None)
    if form_owner is not None and form_owner != user["user_id"]:
        raise HTTPException(403, "You don't have permission to delete this form.")
    await container.form_repo.delete_by_id(form_id)
    logger.info(f"Form deleted: {form_id} by {user['username']}")
    return RedirectResponse(url="/", status_code=303)


@app.delete("/forms/{form_id}", response_class=JSONResponse)
async def delete_form_api(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    form_owner = getattr(form, "owner_id", None)
    if form_owner is not None and form_owner != user["user_id"]:
        raise HTTPException(403, "You don't have permission to delete this form.")
    await container.form_repo.delete_by_id(form_id)
    return JSONResponse({"ok": True})


@app.get("/forms/{form_id}/conversations", response_class=HTMLResponse)
async def list_conversations(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    all_convos = await container.convo_repo.get_by_form_id(form_id)
    convos = [
        c for c in all_convos
        if getattr(c, "owner_id", None) in (None, user["user_id"])
    ]
    return _tmpl("list_conversations.html", request, {"form": form, "convos": convos}, user=user)


@app.get("/forms/{form_id}/enter-conversation", response_class=HTMLResponse)
async def enter_conversation(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("enter_conversation.html", request, {"form": form}, user=user)


@app.get("/forms/{form_id}/live", response_class=HTMLResponse)
async def live_extraction_page(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("static_enter_conversation.html", request, {"form": form}, user=user)


@app.get("/conversations/{convo_id}/edit", response_class=HTMLResponse)
async def edit_conversation_page(request: Request, convo_id: str, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    convo = await _get_convo_for_user(convo_id, user)
    form = await _get_form_for_user(form_id, user)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    if not form:
        raise HTTPException(404, "Form not found")
    raw_text = "\n".join([f"{k.split(' ')[0]}: {v}" for k, v in convo.latest_history.items()])
    return _tmpl("edit_conversation.html", request, {
        "convo": convo, "raw_text": raw_text, "form_id": form_id, "form": form
    }, user=user)


@app.post("/conversations/{convo_id}/update", response_class=RedirectResponse)
async def update_conversation(
    request: Request,
    convo_id: str,
    form_id: str = Form(...),
    new_content: str = Form(...)
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    existing = await _get_convo_for_user(convo_id, user)
    if not existing:
        raise HTTPException(404, "Conversation not found")
    new_history = _parse_conversation_text(new_content)
    if not new_history:
        raise HTTPException(400, "Could not parse updated conversation.")
    new_version_idx = len(existing.versions)
    new_v = ConversationVersion(version_index=new_version_idx, history=new_history)
    existing.versions.append(new_v)
    await container.convo_repo.save(existing)
    return RedirectResponse(url=f"/extract/{form_id}/{convo_id}", status_code=303)


@app.get("/conversations/{convo_id}", response_class=HTMLResponse)
async def view_conversation(request: Request, convo_id: str, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    convo = await _get_convo_for_user(convo_id, user)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    return _tmpl("view_conversation.html", request, {"convo": convo, "form_id": form_id}, user=user)


@app.get("/extract/{form_id}/{convo_id}", response_class=HTMLResponse)
async def run_extraction(request: Request, form_id: str, convo_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    try:
        convo = await _get_convo_for_user(convo_id, user)
        if not convo or not convo.versions:
            raise HTTPException(404, "No conversation history found.")
        latest_version = convo.versions[-1]
        v_idx = latest_version.version_index
        result = await container.pipeline.run(convo_id, form_id, version_index=v_idx)
        latest_version.run_id = result.run_id
        await container.convo_repo.save(convo)
        await _save_output(result)
        json_pretty = json.dumps(result.filled_data, indent=2)
        fields_html = _format_filled_data(result.filled_data)
        return _tmpl("run_extraction.html", request, {
            "form_id": form_id,
            "convo_id": convo_id,
            "convo": convo,
            "fields_html": fields_html,
            "json_pretty": json_pretty,
            "result": result,
            "summary": result.summary,
        }, user=user)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/outputs", response_class=HTMLResponse)
async def view_outputs(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    outputs = await _load_outputs(user["user_id"])
    return _tmpl("view_outputs.html", request, {"outputs": outputs}, user=user)


@app.get("/runs", response_class=HTMLResponse)
async def view_runs(request: Request, limit: int = 20):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    runs = await container.runlog_repo.get_recent(limit)
    return _tmpl("view_runs.html", request, {"runs": runs}, user=user)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    r = await container.runlog_repo.get_by_id(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    ef = getattr(r, "extracted_fields", {}) or {}
    filled_data = ef.get("filled_data", {})
    fields_html = _format_filled_data(filled_data)
    return _tmpl("view_run.html", request, {"run": r, "ef": ef, "fields_html": fields_html}, user=user)


@app.post("/extract/preview", response_class=JSONResponse)
async def preview_extraction(
    request: Request,
    form_id: str = Form(...),
    conversation_text: str = Form(""),
    field_state_json: str = Form("")
):
    user = await _get_current_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    if not conversation_text.strip():
        return JSONResponse({"filled_data": {}, "summary": ""})
    current_field_state: Dict[str, Any] = {}
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
    try:
        result = await _extract_for_conversation_text(form, conversation_text, current_field_state)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/conversations/create", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    form_id: str = Form(...),
    conversation_id: str = Form(""),
    conversation_name: str = Form(""),
    field_overrides_json: str = Form(""),
    conversation_text: str = Form(...)
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    if not conversation_id.strip():
        conversation_id = str(uuid4())[:8]
    cleaned_name = conversation_name.strip()

    reviewed_field_overrides: Dict[str, Any] = {}
    if field_overrides_json.strip():
        try:
            parsed_overrides = json.loads(field_overrides_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid reviewed fields payload: {str(e)}")
        if not isinstance(parsed_overrides, dict):
            raise HTTPException(400, "Reviewed fields payload must be a JSON object.")
        reviewed_field_overrides = parsed_overrides

    conversation_dict = _parse_conversation_text(conversation_text)
    if not conversation_dict:
        raise HTTPException(400, "Could not parse conversation.")

    convo = Conversation(
        conversation_id=conversation_id,
        form_id=form_id,
        conversation_name=cleaned_name,
        versions=[ConversationVersion(version_index=0, history=conversation_dict)],
        owner_id=user["user_id"]
    )
    await container.convo_repo.save(convo)

    try:
        latest_version = convo.versions[-1]
        result = await container.pipeline.run(conversation_id, form_id, version_index=latest_version.version_index)
        if reviewed_field_overrides:
            _apply_field_overrides(result.filled_data, reviewed_field_overrides)
            await container.runlog_repo.update(result.run_id, {
                "extracted_fields": result.model_dump()
            })
        latest_version.run_id = result.run_id
        await container.convo_repo.save(convo)
        await _save_output(result, owner_id=user["user_id"])
    except Exception as e:
        raise HTTPException(500, f"Conversation saved, but extraction failed: {str(e)}")

    return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)


@app.post("/api/live-extract")
async def api_live_extract(request: Request, payload: LiveExtractRequest) -> JSONResponse:
    user = await _get_current_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    form = await _get_form_for_user(payload.form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")
    context = payload.conversation.strip()
    if not context:
        raise HTTPException(400, "Conversation text is empty")
    requests_list: list[ExtractionRequest] = [
        ExtractionRequest(context=context, field_name=field_key, instruction=question)
        for field_key, question in form.fields.items()
    ]
    model = get_live_model()
    answers = await model.extract_batch(requests_list)
    result: Dict[str, str] = {}
    for req, answer in zip(requests_list, answers):
        if answer and answer.strip():
            result[req.field_name] = answer
    return JSONResponse(content=result)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_form_for_user(form_id: str, user: dict):
    """Return form if it belongs to the user or is a global starter."""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        return None
    owner = getattr(form, "owner_id", None)
    if owner is None or owner == user["user_id"]:
        return form
    return None


async def _get_convo_for_user(convo_id: str, user: dict):
    """Return conversation if it belongs to the user or is a global starter."""
    convo = await container.convo_repo.get_by_id(convo_id)
    if not convo:
        return None
    owner = getattr(convo, "owner_id", None)
    if owner is None or owner == user["user_id"]:
        return convo
    return None


def _parse_conversation_text(text: str) -> Dict[str, str]:
    import time
    result = {}
    lines = text.strip().split('\n')
    base_timestamp = int(time.time())
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            key = f"{speaker} {base_timestamp + i}"
            result[key] = message
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
        normalized = "" if value is None else str(value)
        _set_nested_field(target, field_key, normalized)


async def _extract_for_conversation_text(
    form: FormSchema,
    conversation_text: str,
    current_field_state: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    parsed = _parse_conversation_text(conversation_text)
    full_convo = ""
    for speaker, text in parsed.items():
        clean_speaker = " ".join(speaker.split()[:-1]) if " " in speaker else speaker
        full_convo += clean_speaker + ": " + text + "\n"

    seeded_fields: Dict[str, Any] = {}
    for field_key in form.fields.keys():
        if current_field_state and field_key in current_field_state:
            candidate = current_field_state[field_key]
            normalized = "" if candidate is None else str(candidate).strip()
            seeded_fields[field_key] = normalized if normalized else "N/A"
        else:
            seeded_fields[field_key] = "N/A"

    input_str = f"""Extract info from conversation to fill form.\nConversation: {full_convo}Form: {form.name}\nFields: {json.dumps(seeded_fields)}"""

    answers_task = container.pipeline.model.process_extraction_request(input_str)
    summary_task = container.pipeline.summarizer.summarize("\n".join([f"{k}: {v}" for k, v in parsed.items()]))
    answers, summary = await asyncio.gather(answers_task, summary_task)

    filled_data: Dict[str, Any] = {}
    for field_key, value in zip(form.fields.keys(), answers):
        _set_nested_field(filled_data, field_key, value)
    for field_key in form.fields.keys():
        if not _has_nested_field(filled_data, field_key):
            _set_nested_field(filled_data, field_key, "N/A")

    return {"filled_data": filled_data, "summary": summary}


def _format_filled_data(data: Dict[str, Any], prefix: str = "") -> str:
    html = ""
    for k, v in data.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            html += _format_filled_data(v, key)
        else:
            html += f'<div class="field"><span class="field-name">{key}:</span> <span class="field-value">{v}</span></div>'
    return html


async def _save_output(result: ExtractionResult, owner_id: str = None):
    collection = container.convo_repo.db.outputs
    doc = result.model_dump()
    if owner_id:
        doc["owner_id"] = owner_id
    await collection.insert_one(doc)
    logger.info("Extraction result saved to Atlas.")


async def _load_outputs(user_id: str = None) -> List[Dict]:
    collection = container.convo_repo.db.outputs
    query = {}
    if user_id:
        # Show user's outputs + outputs with no owner (from global starters)
        query = {"$or": [{"owner_id": user_id}, {"owner_id": {"$exists": False}}, {"owner_id": None}]}
    cursor = collection.find(query).sort("_id", -1)
    return await cursor.to_list(length=100)