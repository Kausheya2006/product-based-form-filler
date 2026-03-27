import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from typing import List, Dict, Any

from ..domain.domain import Conversation, FormSchema, ConversationVersion
from ..domain.speakers import render_history_for_model
from .dependencies import container, Container

from pydantic import BaseModel as PydanticBaseModel

from .helpers import (
    SESSION_COOKIE, SESSION_MAX_AGE, ADMIN_USERNAME, templates, logger,
    _hash_password, _verify_password, _make_session_token, _is_admin,
    _get_current_user, _user_repo, _tmpl, _validate_username, seed_data,
    _is_global, _can_write_form, _can_write_convo,
    _get_form_for_user, _get_convo_for_user, _parse_conversation_text, _build_schema_from_pairs,
    _apply_field_overrides, _extract_for_conversation_text, _format_filled_data, _save_output,
    _load_outputs,
)

from .collab_ws import router as collab_router

class LiveExtractRequest(PydanticBaseModel):
    form_id: str
    conversation: str


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
    return f"Speaker: {cleaned}"


async def _persist_conversation_and_extract(
    *,
    form_id: str,
    conversation_text: str,
    owner_id: str,
    conversation_id: str = "",
    conversation_name: str = "",
    reviewed_field_overrides: Dict[str, Any] | None = None,
    version_metadata: Dict[str, Any] | None = None,
) -> str:
    if not conversation_id.strip():
        conversation_id = str(uuid4())[:8]

    conversation_dict = _parse_conversation_text(conversation_text)
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

    result = await container.pipeline.run(conversation_id, form_id, version_index=0)
    if reviewed_field_overrides:
        _apply_field_overrides(result.filled_data, reviewed_field_overrides)
        await container.runlog_repo.update(result.run_id, {"extracted_fields": result.model_dump()})

    latest_version = convo.versions[-1]
    latest_version.run_id = result.run_id
    await container.convo_repo.save(convo)
    await _save_output(result, owner_id=owner_id)
    return conversation_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    Container.initialize()
    await container.runlog_repo.ensure_indexes()
    await _user_repo().create_index("username", unique=True)
    await _user_repo().create_index("email", unique=True)
    await seed_data()
    logger.info("Application started.")
    yield

app = FastAPI(title="ProductLabs Form Filler", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/interface/static"), name="static")
app.include_router(collab_router)

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {
        "request": request, "error": None, "prefill": {}
    })

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
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
    if await _user_repo().find_one({"username": username}):
        return _err("That username is already taken.")
    if await _user_repo().find_one({"email": email.lower()}):
        return _err("An account with that email already exists.")

    user_id = str(uuid4())
    await _user_repo().insert_one({
        "user_id": user_id,
        "username": username,
        "email": email.lower(),
        "password_hash": _hash_password(password),
        "role": "user",
        "created_at": datetime.utcnow(),
    })
    logger.info(f"New user registered: {username} ({user_id})")

    token = _make_session_token(user_id)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax")
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, registered: str = ""):
    return templates.TemplateResponse(request, "login.html", {
        "request": request,
        "error": None,
        "success": "Account created! Please sign in." if registered == "1" else None,
        "prefill_username": "",
    })

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = await _user_repo().find_one({"username": username})
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "error": "Incorrect username or password.",
            "success": None,
            "prefill_username": username,
        })

    token = _make_session_token(user["user_id"])
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax")
    logger.info(f"User logged in: {username} (role={user.get('role', 'user')})")
    return response

@app.post("/logout", response_class=RedirectResponse)
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return _tmpl("profile.html", request, {
        "username_error": None, "username_success": None,
        "password_error": None, "password_success": None,
    }, user=user)

@app.post("/profile/change-username", response_class=HTMLResponse)
async def change_username(request: Request, new_username: str = Form(...)):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    new_username = new_username.strip()
    ctx = {"username_error": None, "username_success": None,
           "password_error": None, "password_success": None}

    if error := _validate_username(new_username):
        ctx["username_error"] = error
        return _tmpl("profile.html", request, ctx, user=user)
    if new_username == user["username"]:
        ctx["username_error"] = "That's already your current username."
        return _tmpl("profile.html", request, ctx, user=user)
    if new_username.lower() == ADMIN_USERNAME.lower() and not _is_admin(user):
        ctx["username_error"] = "That username is reserved."
        return _tmpl("profile.html", request, ctx, user=user)
    if await _user_repo().find_one({"username": new_username}):
        ctx["username_error"] = "That username is already taken."
        return _tmpl("profile.html", request, ctx, user=user)

    await _user_repo().update_one(
        {"user_id": user["user_id"]}, {"$set": {"username": new_username}}
    )
    user["username"] = new_username
    ctx["username_success"] = "Username updated successfully."
    return _tmpl("profile.html", request, ctx, user=user)

@app.post("/profile/change-password", response_class=HTMLResponse)
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_new_password: str = Form(...),
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    ctx = {"username_error": None, "username_success": None,
           "password_error": None, "password_success": None}

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
        {"$set": {"password_hash": _hash_password(new_password)}},
    )
    ctx["password_success"] = "Password updated successfully."
    return _tmpl("profile.html", request, ctx, user=user)

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        raise HTTPException(403, "Admin access required.")
    all_users = await _user_repo().find({}).sort("created_at", 1).to_list(length=500)
    return _tmpl("admin_users.html", request, {"all_users": all_users}, user=user)

@app.post("/admin/users/{target_user_id}/set-role", response_class=RedirectResponse)
async def admin_set_role(request: Request, target_user_id: str, role: str = Form(...)):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        raise HTTPException(403, "Admin access required.")
    if role not in ("admin", "user"):
        raise HTTPException(400, "Invalid role.")
    if target_user_id == user["user_id"] and role != "admin":
        raise HTTPException(400, "You cannot remove your own admin role.")
    await _user_repo().update_one({"user_id": target_user_id}, {"$set": {"role": role}})
    logger.info(f"Admin {user['username']} set role={role} for user_id={target_user_id}")
    return RedirectResponse(url="/admin/users", status_code=303)

@app.post("/admin/users/{target_user_id}/delete", response_class=RedirectResponse)
async def admin_delete_user(request: Request, target_user_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        raise HTTPException(403, "Admin access required.")
    if target_user_id == user["user_id"]:
        raise HTTPException(400, "You cannot delete your own account.")
    await _user_repo().delete_one({"user_id": target_user_id})
    logger.info(f"Admin {user['username']} deleted user_id={target_user_id}")
    return RedirectResponse(url="/admin/users", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    all_forms = await container.form_repo.get_all()
    username  = user["username"]
    is_admin  = _is_admin(user)

    if is_admin:
        visible_forms = all_forms
    else:
        visible_forms = [
            f for f in all_forms
            if getattr(f, "visibility", None) == "global"
            or getattr(f, "owner_id", None) is None          # legacy seed data — treat as global
            or getattr(f, "owner_id", None) == user["user_id"]  # personal or collab owned by this user
            or (getattr(f, "visibility", "") == "collaborative"
                and username in getattr(f, "collaborators", []))
        ]

    deletable_form_ids = {f.id for f in visible_forms if getattr(f, "owner_id", None) == user["user_id"]}
    form_owners = {}
    if is_admin:
        all_user_ids = {f.owner_id for f in visible_forms if f.owner_id}
        if all_user_ids:
            docs = await _user_repo().find({"user_id": {"$in": list(all_user_ids)}}).to_list(length=500)
            form_owners = {d["user_id"]: d["username"] for d in docs}

    return _tmpl("home.html", request, {
        "forms":             visible_forms,
        "deletable_form_ids": deletable_form_ids,
        "form_owners":       form_owners,
    }, user=user)

@app.post("/forms", response_class=RedirectResponse)
async def create_form(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form_data = await request.form()
    form_name   = form_data.get("form_name", "").strip()
    description = form_data.get("form_description", "").strip()
    visibility  = form_data.get("visibility", "personal")  # "personal"|"global"|"collaborative"

    # Only admins may create global forms; silently demote to personal for everyone else.
    if visibility == "global" and not _is_admin(user):
        visibility = "personal"

    collaborators = [
        col.strip()
        for col in form_data.getlist("collaborator[]")
        if col.strip()
    ]

    field_names = form_data.getlist("field_name[]")
    field_types = form_data.getlist("field_type[]")
    fields = {k: v for k, v in zip(field_names, field_types) if k.strip()}
    if not fields:
        raise HTTPException(400, "At least one valid field is required.")

    new_form = FormSchema(**{
        "form_id":   str(uuid4()),
        "form_name": form_name,
        "schema":    fields,
    },
        description=description,
        owner_id=user["user_id"],
        visibility=visibility,
        collaborators=collaborators if visibility == "collaborative" else [],
    )
    await container.form_repo.save(new_form)
    return RedirectResponse(url="/", status_code=303)

@app.get("/forms/new", response_class=HTMLResponse)
async def new_form_page(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return _tmpl("create_form.html", request, {"is_admin": _is_admin(user)}, user=user)

@app.get("/forms/{form_id}", response_class=HTMLResponse)
async def view_form(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        # Fallback: collaborators on a collaborative form may not be returned
        # by _get_form_for_user — fetch directly and check.
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
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
    # Regular users can only edit forms they own; global forms are read-only.
    # They are still allowed to reach the edit page to use "Save As New Form".
    can_save_in_place = _can_write_form(form, user)
    return _tmpl("edit_form.html", request, {"form": form, "can_save_in_place": can_save_in_place}, user=user)

@app.post("/forms/{form_id}/edit", response_class=RedirectResponse)
async def save_form_edits(
    request: Request,
    form_id: str,
    form_name: str = Form(...),
    form_description: str = Form(""),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_instruction: List[str] = Form(..., alias="field_instruction[]"),
    save_mode: str = Form("save"),
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    existing = await _get_form_for_user(form_id, user)
    if not existing:
        raise HTTPException(404, "Form not found")

    # Regular users cannot overwrite a global or another user's form in-place.
    if save_mode == "save" and not _can_write_form(existing, user):
        raise HTTPException(403, "You can only save a copy of this form using 'Save As New Form'.")

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

    await container.form_repo.save(FormSchema(
        form_id=target_form_id,
        form_name=cleaned_name,
        description=cleaned_description,
        schema=schema_dict,
        owner_id=owner,
    ))
    return RedirectResponse(url=f"/forms/{target_form_id}", status_code=303)

@app.post("/forms/{form_id}/delete", response_class=RedirectResponse)
async def delete_form(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    if not _can_write_form(form, user):
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
    if not _can_write_form(form, user):
        raise HTTPException(403, "You don't have permission to delete this form.")
    await container.form_repo.delete_by_id(form_id)
    return JSONResponse({"ok": True})

@app.get("/forms/{form_id}/collab", response_class=HTMLResponse)
async def enter_collab_conversation(request: Request, form_id: str):
    """Launches the collaborative live-entry page."""
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    room_id = f"form-{form_id}"
    return _tmpl("enter_conversation_collab.html", request, {
        "form":         form,
        "room_id":      room_id,
        "current_user": user["username"],
    }, user=user)

@app.get("/forms/{form_id}/conversations", response_class=HTMLResponse)
async def list_conversations(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
    if not form:
        raise HTTPException(404, "Form not found")
    all_convos = await container.convo_repo.get_by_form_id(form_id)
    if _is_admin(user):
        convos = all_convos
    else:
        convos = [c for c in all_convos if getattr(c, "owner_id", None) in (None, user["user_id"])]
    return _tmpl("list_conversations.html", request, {"form": form, "convos": convos}, user=user)

@app.get("/forms/{form_id}/enter-conversation", response_class=HTMLResponse)
async def enter_conversation(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
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
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("static_enter_conversation.html", request, {"form": form}, user=user)


@app.get("/forms/{form_id}/asr", response_class=HTMLResponse)
async def asr_extraction_page(request: Request, form_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    form = await _get_form_for_user(form_id, user)
    if not form:
        raw_form = await container.form_repo.get_by_id(form_id)
        if raw_form and getattr(raw_form, "visibility", "") == "collaborative" \
                and user["username"] in getattr(raw_form, "collaborators", []):
            form = raw_form
    if not form:
        raise HTTPException(404, "Form not found")
    return _tmpl("asr_enter_conversation.html", request, {
        "form": form,
        "input_languages": SUPPORTED_INPUT_LANGUAGES,
    }, user=user)

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
    raw_text = render_history_for_model(convo.latest_history)
    return _tmpl("edit_conversation.html", request, {
        "convo": convo, "raw_text": raw_text, "form_id": form_id, "form": form,
    }, user=user)

@app.post("/conversations/{convo_id}/update", response_class=RedirectResponse)
async def update_conversation(
    request: Request,
    convo_id: str,
    form_id: str = Form(...),
    new_content: str = Form(...),
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
    new_v = ConversationVersion(version_index=len(existing.versions), history=new_history)
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

@app.post("/conversations/create", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    form_id: str = Form(...),
    conversation_id: str = Form(""),
    conversation_name: str = Form(""),
    field_overrides_json: str = Form(""),
    conversation_text: str = Form(...),
):
    user = await _get_current_user(request)
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

    try:
        conversation_id = await _persist_conversation_and_extract(
            form_id=form_id,
            conversation_text=conversation_text,
            owner_id=user["user_id"],
            conversation_id=conversation_id,
            conversation_name=conversation_name,
            reviewed_field_overrides=reviewed_field_overrides,
            version_metadata={"source_mode": "text"},
        )
    except Exception as e:
        raise HTTPException(500, f"Conversation saved, but extraction failed: {str(e)}")

    return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)


@app.post("/conversations/create-asr", response_class=HTMLResponse)
async def create_conversation_asr(
    request: Request,
    form_id: str = Form(...),
    input_language: str = Form("en"),
    conversation_id: str = Form(""),
    conversation_name: str = Form(""),
    conversation_text: str = Form(""),
    audio_file: UploadFile | None = File(default=None),
):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    form = await _get_form_for_user(form_id, user)
    if not form:
        raise HTTPException(404, "Form not found")

    transcript_text = ""
    if audio_file is not None and (audio_file.filename or "").strip():
        raw_audio = await audio_file.read()
        if not raw_audio:
            raise HTTPException(400, "Uploaded audio file is empty.")
        transcript_text = await container.asr_transcriber.transcribe_to_text(
            audio_bytes=raw_audio,
            filename=audio_file.filename,
            input_language=input_language,
        )
    elif conversation_text.strip():
        transcript_text = conversation_text
    else:
        raise HTTPException(400, "Please upload an audio file or record audio.")

    translated_text = await container.translator.translate_to_english(transcript_text, input_language)
    conversation_payload = _transcript_to_conversation_text(translated_text)
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
        )
    except Exception as e:
        raise HTTPException(500, f"Conversation saved, but extraction failed: {str(e)}")

    return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)

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
        result = await container.pipeline.run(
            convo_id, form_id, version_index=latest_version.version_index
        )
        latest_version.run_id = result.run_id
        await container.convo_repo.save(convo)
        await _save_output(result, owner_id=user["user_id"])
        return _tmpl("run_extraction.html", request, {
            "form_id": form_id,
            "convo_id": convo_id,
            "convo": convo,
            "fields_html": _format_filled_data(result.filled_data),
            "json_pretty": json.dumps(result.filled_data, indent=2),
            "result": result,
            "summary": result.summary,
        }, user=user)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/extract/preview", response_class=JSONResponse)
async def preview_extraction(
    request: Request,
    form_id: str = Form(...),
    conversation_text: str = Form(""),
    field_state_json: str = Form(""),
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
    logger.info("[PreviewExtract] form_id=%s", form_id)
    logger.info("[PreviewExtract] conversation_text=%s", conversation_text)
    logger.info("[PreviewExtract] current_field_state=%s", current_field_state)
    try:
        extraction = await _extract_for_conversation_text(form, conversation_text, current_field_state)
        logger.info("[PreviewExtract] response=%s", extraction)
        return JSONResponse(extraction)
    except Exception as e:
        logger.exception("[PreviewExtract] failed")
        raise HTTPException(500, str(e))


@app.post("/stt/transcribe", response_class=JSONResponse)
async def transcribe_live_audio(
    request: Request,
    input_language: str = Form("en"),
    audio_file: UploadFile | None = File(default=None),
):
    user = await _get_current_user(request)
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
        recoverable_markers = (
            "audio preprocessing failed",
            "audio decode failed",
            "speechrecognition decode failed",
            "unknown format",
            "cannot read",
            "file does not start",
        )
        if any(marker in message.lower() for marker in recoverable_markers):
            return JSONResponse({
                "text": "",
                "raw_text": "",
                "warning": message,
            })
        raise HTTPException(500, message) from exc
    except Exception as exc:
        logger.exception("[STT] Unexpected transcription failure")
        raise HTTPException(500, f"STT failed: {str(exc)}") from exc

    try:
        translated_text = await container.translator.translate_to_english(transcript_text, input_language)
    except Exception as exc:
        logger.exception("[STT] Translation step failed")
        raise HTTPException(500, f"STT translation failed: {str(exc)}") from exc
    return JSONResponse({
        "text": translated_text.strip(),
        "raw_text": transcript_text.strip(),
    })

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
    extraction = await _extract_for_conversation_text(form, context)
    return JSONResponse(content=extraction.get("filled_data", {}))

@app.get("/outputs", response_class=HTMLResponse)
async def view_outputs(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return _tmpl("view_outputs.html", request, {"outputs": await _load_outputs(user)}, user=user)

@app.get("/runs", response_class=HTMLResponse)
async def view_runs(request: Request, limit: int = 50):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        raise HTTPException(403, "Run logs are only available to administrators.")
    runs = await container.runlog_repo.get_recent(limit)
    return _tmpl("view_runs.html", request, {"runs": runs}, user=user)

@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        raise HTTPException(403, "Run logs are only available to administrators.")
    r = await container.runlog_repo.get_by_id(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    ef = getattr(r, "extracted_fields", {}) or {}
    return _tmpl("view_run.html", request, {
        "run": r,
        "ef": ef,
        "fields_html": _format_filled_data(ef.get("filled_data", {})),
    }, user=user)