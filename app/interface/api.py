import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from typing import List, Dict, Any

from ..domain.domain import Conversation, FormSchema, ConversationVersion, ExtractionResult
from ..domain.speakers import render_history_for_model
from .dependencies import container, Container

from pydantic import BaseModel as PydanticBaseModel

from .helpers import (
    AuthService, UserRepository, AccessPolicy,
    FormQueryService, ConvoQueryService,
    ConversationParser, SchemaBuilder, FieldMerger,
    ExtractionService, OutputRepository, TemplateRenderer,
    SESSION_COOKIE, SESSION_MAX_AGE, ADMIN_USERNAME, templates, logger,
    _hash_password, _verify_password, _make_session_token, _is_admin,
    _get_current_user, _user_repo, _tmpl, _validate_username,
    _is_global, _can_write_form, _can_write_convo,
    _get_form_for_user, _get_convo_for_user,
    _parse_conversation_text, _build_schema_from_pairs,
    _apply_field_overrides, _extract_for_conversation_text,
    _format_filled_data, _merge_display_fields, _save_output,
    _load_outputs, _load_output_by_run_id,
    AuthHandler, AdminHandler, FormHandler, ConversationHandler, ExtractionHandler, ASRHandler, OutputHandler,_diarized_to_conversation_text
)

from .collab_ws import router as collab_router

class LiveExtractRequest(PydanticBaseModel):
    form_id: str
    conversation: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    Container.initialize()
    await container.runlog_repo.ensure_indexes()
    await UserRepository.collection().create_index("username", unique=True)
    await UserRepository.collection().create_index("email",    unique=True)
    logger.info("Application started.")
    yield


app = FastAPI(title="ProductLabs Form Filler", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/interface/static"), name="static")
app.include_router(collab_router)

@app.get("/register",  response_class=HTMLResponse)
async def register_page(request: Request): return await AuthHandler.register_page(request)

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, email: str = Form(...), username: str = Form(...),
                   password: str = Form(...), confirm_password: str = Form(...)):
    return await AuthHandler.register(request, email, username, password, confirm_password)

@app.get("/login",  response_class=HTMLResponse)
async def login_page(request: Request, registered: str = ""): return await AuthHandler.login_page(request, registered)

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    return await AuthHandler.login(request, username, password)

@app.post("/logout", response_class=RedirectResponse)
async def logout(): return await AuthHandler.logout()

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request): return await AuthHandler.profile_page(request)

@app.post("/profile/change-username", response_class=HTMLResponse)
async def change_username(request: Request, new_username: str = Form(...)):
    return await AuthHandler.change_username(request, new_username)

@app.post("/profile/change-password", response_class=HTMLResponse)
async def change_password(request: Request, old_password: str = Form(...),
                          new_password: str = Form(...), confirm_new_password: str = Form(...)):
    return await AuthHandler.change_password(request, old_password, new_password, confirm_new_password)

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request): return await AdminHandler.list_users(request)

@app.post("/admin/users/{target_user_id}/set-role", response_class=RedirectResponse)
async def admin_set_role(request: Request, target_user_id: str, role: str = Form(...)):
    return await AdminHandler.set_role(request, target_user_id, role)

@app.post("/admin/users/{target_user_id}/delete", response_class=RedirectResponse)
async def admin_delete_user(request: Request, target_user_id: str):
    return await AdminHandler.delete_user(request, target_user_id)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request): return await FormHandler.home(request)

@app.get("/forms/new", response_class=HTMLResponse)
async def new_form_page(request: Request): return await FormHandler.new_form_page(request)

@app.post("/forms", response_class=RedirectResponse)
async def create_form(
    request: Request,
    form_name: str        = Form(...),
    form_description: str = Form(""),
    visibility: str       = Form("personal"),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_type: List[str] = Form(..., alias="field_type[]"),
    collaborator: List[str] = Form([], alias="collaborator[]"),
):
    return await FormHandler.create_form(
        request, form_name, form_description, visibility,
        field_name, field_type, collaborator,
    )

@app.get("/forms/{form_id}", response_class=HTMLResponse)
async def view_form(request: Request, form_id: str): return await FormHandler.view_form(request, form_id)

@app.get("/forms/{form_id}/edit", response_class=HTMLResponse)
async def edit_form(request: Request, form_id: str): return await FormHandler.edit_form_page(request, form_id)

@app.post("/forms/{form_id}/edit", response_class=RedirectResponse)
async def save_form_edits(
    request: Request, form_id: str,
    form_name: str        = Form(...),
    form_description: str = Form(""),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_instruction: List[str] = Form(..., alias="field_instruction[]"),
    save_mode: str        = Form("save"),
):
    return await FormHandler.save_form_edits(
        request, form_id, form_name, form_description, field_name, field_instruction, save_mode
    )

@app.post("/forms/{form_id}/delete", response_class=RedirectResponse)
async def delete_form(request: Request, form_id: str): return await FormHandler.delete_form(request, form_id)

@app.delete("/forms/{form_id}", response_class=JSONResponse)
async def delete_form_api(request: Request, form_id: str): return await FormHandler.delete_form_api(request, form_id)

@app.get("/forms/{form_id}/enter-conversation", response_class=HTMLResponse)
async def enter_conversation(request: Request, form_id: str):
    return await ConversationHandler.enter_conversation_page(request, form_id)

@app.get("/forms/{form_id}/collab", response_class=HTMLResponse)
async def enter_collab_conversation(request: Request, form_id: str):
    return await ConversationHandler.enter_collab_conversation(request, form_id)

@app.get("/forms/{form_id}/conversations", response_class=HTMLResponse)
async def list_conversations(request: Request, form_id: str):
    return await ConversationHandler.list_conversations(request, form_id)

@app.get("/conversations/{convo_id}", response_class=HTMLResponse)
async def view_conversation(request: Request, convo_id: str, form_id: str):
    return await ConversationHandler.view_conversation(request, convo_id, form_id)

@app.get("/conversations/{convo_id}/edit", response_class=HTMLResponse)
async def edit_conversation_page(request: Request, convo_id: str, form_id: str):
    return await ConversationHandler.edit_conversation_page(request, convo_id, form_id)

@app.post("/conversations/{convo_id}/update", response_class=RedirectResponse)
async def update_conversation(request: Request, convo_id: str,
                               form_id: str = Form(...), new_content: str = Form(...)):
    return await ConversationHandler.update_conversation(request, convo_id, form_id, new_content)

@app.post("/conversations/create", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    form_id: str               = Form(...),
    conversation_id: str       = Form(""),
    conversation_name: str     = Form(""),
    field_overrides_json: str  = Form(""),
    accepted_new_fields_json: str = Form(""),
    conversation_text: str     = Form(...),
    extract: bool              = Form(True),
):
    return await ConversationHandler.create_conversation(
        request, form_id, conversation_id, conversation_name,
        field_overrides_json, accepted_new_fields_json, conversation_text, extract,
    )

@app.post("/asr/diarize-preview", response_class=JSONResponse)
async def asr_diarize_preview(
    request: Request,
    input_language: str = Form("en"),
    num_speakers: int = Form(2),
    audio_file: UploadFile | None = File(default=None),
):
    """
    Run speaker diarization on an uploaded audio file and return a
    speaker-labelled conversation transcript for preview before final submission.
    """
    user = await _get_current_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    if audio_file is None or not (audio_file.filename or "").strip():
        raise HTTPException(400, "Audio file is required.")

    raw_audio = await audio_file.read()
    if not raw_audio:
        raise HTTPException(400, "Uploaded audio file is empty.")

    num_speakers = max(1, int(num_speakers))

    try:
        diarized_turns = await container.diarizer.diarize(
            audio_bytes=raw_audio,
            filename=audio_file.filename,
            num_speakers=num_speakers,
            input_language=input_language,
        )
    except RuntimeError as exc:
        logger.warning("[Diarize] Runtime error: %s", exc)
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:
        logger.exception("[Diarize] Unexpected failure")
        raise HTTPException(500, f"Diarization failed: {str(exc)}") from exc

    diarized_text = _diarized_to_conversation_text(diarized_turns)
    raw_text = " ".join(t.get("text", "") for t in diarized_turns).strip()

    return JSONResponse({
        "diarized_text": diarized_text,
        "raw_text": raw_text,
        "turns": diarized_turns,
    })

@app.get("/extract/{form_id}/{convo_id}", response_class=HTMLResponse)
async def run_extraction(request: Request, form_id: str, convo_id: str):
    return await ExtractionHandler.run_extraction(request, form_id, convo_id)

@app.post("/extract/preview", response_class=JSONResponse)
async def preview_extraction(
    request: Request,
    form_id: str               = Form(...),
    conversation_text: str     = Form(""),
    field_state_json: str      = Form(""),
    accepted_new_fields_json: str = Form(""),
):
    return await ExtractionHandler.preview_extraction(
        request, form_id, conversation_text, field_state_json, accepted_new_fields_json
    )

@app.post("/api/live-extract")
async def api_live_extract(request: Request, payload: LiveExtractRequest):
    return await ExtractionHandler.api_live_extract(request, payload)

@app.get("/forms/{form_id}/live", response_class=HTMLResponse)
async def live_extraction_page(request: Request, form_id: str):
    return await ExtractionHandler.live_extraction_page(request, form_id)

@app.get("/forms/{form_id}/asr", response_class=HTMLResponse)
async def asr_extraction_page(request: Request, form_id: str):
    return await ASRHandler.asr_extraction_page(request, form_id)

@app.post("/conversations/create-asr", response_class=HTMLResponse)
async def create_conversation_asr(
    request: Request,
    form_id: str               = Form(...),
    input_language: str        = Form("en"),
    conversation_id: str       = Form(""),
    conversation_name: str     = Form(""),
    conversation_text: str     = Form(""),
    translated_text_override: str = Form(""),
    raw_transcript_override: str  = Form(""),
    num_speakers: int = Form(0),
    audio_file: UploadFile | None = File(default=None),
):
    return await ASRHandler.create_conversation_asr(
        request, form_id, input_language, conversation_id, conversation_name,
        conversation_text, translated_text_override, raw_transcript_override, num_speakers, audio_file,
    )

@app.post("/stt/transcribe", response_class=JSONResponse)
async def transcribe_live_audio(
    request: Request,
    input_language: str        = Form("en"),
    audio_file: UploadFile | None = File(default=None),
):
    return await ASRHandler.transcribe_live_audio(request, input_language, audio_file)

@app.post("/asr/translate-preview", response_class=JSONResponse)
async def asr_translate_preview(
    request: Request,
    input_language: str        = Form("en"),
    audio_file: UploadFile | None = File(default=None),
):
    return await ASRHandler.asr_translate_preview(request, input_language, audio_file)

@app.get("/outputs", response_class=HTMLResponse)
async def view_outputs(request: Request): return await OutputHandler.view_outputs(request)

@app.get("/outputs/{run_id}", response_class=HTMLResponse)
async def view_output_detail(request: Request, run_id: str):
    return await OutputHandler.view_output_detail(request, run_id)

@app.get("/runs", response_class=HTMLResponse)
async def view_runs(request: Request, limit: int = 50): return await OutputHandler.view_runs(request, limit)

@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str): return await OutputHandler.view_run(request, run_id)