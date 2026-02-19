"""FastAPI Interface - Routes only, DI handled by dependencies.py"""
import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime

from ..domain.domain import Conversation, FormSchema, ExtractionResult, ConversationVersion
from .dependencies import container, Container

from pydantic import BaseModel as PydanticBaseModel
from ..infrastructure.ai.local_model import LocalHuggingFaceModel
from ..domain.domain import ExtractionRequest

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

def _convert_mongo_types(obj):
    """Recursively convert Mongo Extended JSON types to native Python types.
    Handles {'$date': ISOString} -> datetime and {'$oid': id} -> str.
    """
    if isinstance(obj, dict):
        # Mongo date format: {'$date': '2026-02-09T16:44:08.176Z'}
        if '$date' in obj:
            s = obj['$date']
            # some exports wrap dates in nested structures
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
    """Seed MongoDB from JSON files"""
    if os.path.exists("data/conversations.json"):
        with open("data/conversations.json", "r") as f:
            for raw in json.load(f):
                data = _convert_mongo_types(raw)
                # ensure ids are strings
                data["conversation_id"] = str(data.get("conversation_id", ""))

                history = data.pop("history", data.pop("conversation", None))
                if history and not data.get("versions"):
                    data["versions"] = [ConversationVersion(version_index=0, history=history).model_dump()]

                await container.convo_repo.save(Conversation(**data))
        logger.info("Conversations seeded.")
    if os.path.exists("data/forms.json"):
        with open("data/forms.json", "r") as f:
            for raw in json.load(f):
                data = _convert_mongo_types(raw)
                data["form_id"] = str(data.get("form_id", ""))
                await container.form_repo.save(FormSchema(**data))
        logger.info("Forms seeded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    Container.initialize()
    await container.runlog_repo.ensure_indexes()  
    await seed_data()
    logger.info("Application started.")
    yield

app = FastAPI(title="ProductLabs Form Filler", lifespan=lifespan)

# serve static assets (JS/CSS/images)
app.mount("/static", StaticFiles(directory="src/interface/static"), name="static")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - list all forms"""
    forms = await container.form_repo.get_all()
    return templates.TemplateResponse("home.html", {"request": request, "forms": forms})

@app.post("/forms", response_class=RedirectResponse)
async def handle_create_form(
    request: Request,
    form_name: str = Form(...),
    form_description: str = Form(""),
    field_name: List[str] = Form(..., alias="field_name[]"),
    field_type: List[str] = Form(..., alias="field_type[]")
):
    """Saves a new form schema to MongoDB and redirects home"""
    form_id = str(uuid4())[:8]
    
    # Map names to types for the 'schema' field in MongoDB
    schema_dict = {name: ftype for name, ftype in zip(field_name, field_type) if name.strip()}
    
    new_form = FormSchema(
        form_id=form_id,
        form_name=form_name,
        description=form_description,
        schema=schema_dict
    )
    
    # Persist to Mongo via the repository
    await container.form_repo.save(new_form)
    logger.info(f"Form {form_id} created in MongoDB.")
    
    # Return to the home page list
    return RedirectResponse(url="/", status_code=303)


@app.get("/forms/new", response_class=HTMLResponse)
async def create_form(request: Request):
    """Create a new form"""
    return templates.TemplateResponse("create_form.html", {"request": request})


@app.get("/forms/{form_id}", response_class=HTMLResponse)
async def view_form(request: Request, form_id: str):
    """Display form details"""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    return templates.TemplateResponse("view_form.html", {"request": request, "form": form})

@app.get("/forms/{form_id}/conversations", response_class=HTMLResponse)
async def list_conversations(request: Request, form_id: str):
    """List conversations linked to a form"""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    
    convos = await container.convo_repo.get_by_form_id(form_id)
    
    return templates.TemplateResponse("list_conversations.html", {"request": request, "form": form, "convos": convos})

@app.get("/forms/{form_id}/enter-conversation", response_class=HTMLResponse)
async def enter_conversation(request: Request, form_id: str):
    """List conversations linked to a form"""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
        
    return templates.TemplateResponse("enter_conversation.html", {"request": request, "form": form})

def _parse_conversation_text(text: str) -> Dict[str, str]:
    """Parse conversation text into dict with timestamp keys"""
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
    """Populate nested objects from dotted keys."""
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

async def _extract_for_conversation_text(form: FormSchema, conversation_text: str) -> Dict[str, Any]:
    """Run extraction directly from raw conversation text using full-process prompt."""
    parsed = _parse_conversation_text(conversation_text)
    full_convo = ""
    for speaker, text in parsed.items():
        clean_speaker = " ".join(speaker.split()[:-1]) if " " in speaker else speaker
        full_convo += clean_speaker + ": " + text + "\n"

    # Keep exact prompt shape used in full_process mode.
    empty_fields = {k: "N/A" for k in form.fields.keys()}
    input_str = f"""Extract info from conversation to fill form.\nConversation: {full_convo}Form: {form.name}\nFields: {json.dumps(empty_fields)}"""

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

@app.post("/extract/preview", response_class=JSONResponse)
async def preview_extraction(form_id: str = Form(...), conversation_text: str = Form("")):
    """Preview extraction in-page without saving runs/conversations."""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")

    if not conversation_text.strip():
        return JSONResponse({"filled_data": {}, "summary": ""})

    try:
        result = await _extract_for_conversation_text(form, conversation_text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/conversations/create", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    form_id: str = Form(...),
    conversation_id: str = Form(""), 
    conversation_text: str = Form(...)
):
    """Initializes a new conversation with Version 0 and persists extraction output."""
    if not conversation_id.strip():
        conversation_id = str(uuid4())[:8]
    
    conversation_dict = _parse_conversation_text(conversation_text)
    
    if not conversation_dict:
        raise HTTPException(400, "Could not parse conversation.")
    
    # Initialize with the first version
    convo = Conversation(
        conversation_id=conversation_id,
        form_id=form_id,
        versions=[ConversationVersion(version_index=0, history=conversation_dict)]
    )
    await container.convo_repo.save(convo)

    # Run extraction immediately on save, mirroring /extract behavior for persistence.
    try:
        latest_version = convo.versions[-1]
        result = await container.pipeline.run(conversation_id, form_id, version_index=latest_version.version_index)
        latest_version.run_id = result.run_id
        await container.convo_repo.save(convo)
        await _save_output(result)
    except Exception as e:
        raise HTTPException(500, f"Conversation saved, but extraction failed: {str(e)}")

    return RedirectResponse(url=f"/forms/{form_id}/conversations", status_code=303)

@app.get("/conversations/{convo_id}/edit", response_class=HTMLResponse)
async def edit_conversation_page(request: Request, convo_id: str, form_id: str):
    convo = await container.convo_repo.get_by_id(convo_id)
    form = await container.form_repo.get_by_id(form_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    if not form:
        raise HTTPException(404, "Form not found")
    # Convert dict history back to raw text for the textarea
    raw_text = "\n".join([f"{k.split(' ')[0]}: {v}" for k, v in convo.latest_history.items()])
    return templates.TemplateResponse("edit_conversation.html", {
        "request": request, 
        "convo": convo, 
        "raw_text": raw_text, 
        "form_id": form_id,
        "form": form
    })

@app.post("/conversations/{convo_id}/update", response_class=RedirectResponse)
async def update_conversation(
    convo_id: str,
    form_id: str = Form(...),
    new_content: str = Form(...)
):
    existing = await container.convo_repo.get_by_id(convo_id)
    if not existing:
        raise HTTPException(404, "Conversation not found")
    
    # Parse new content and determine version index
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
    """Display conversation details"""
    convo = await container.convo_repo.get_by_id(convo_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    
    return templates.TemplateResponse("view_conversation.html", {"request": request, "convo": convo, "form_id": form_id})

@app.get("/extract/{form_id}/{convo_id}", response_class=HTMLResponse)
async def run_extraction(request: Request, form_id: str, convo_id: str):
    """Run pipeline and display result"""
    try:
        convo = await container.convo_repo.get_by_id(convo_id)
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
        return templates.TemplateResponse(
            "run_extraction.html",
            {
                "request": request,
                "form_id": form_id,
                "convo_id": convo_id,
                "convo": convo,
                "fields_html": fields_html,
                "json_pretty": json_pretty,
                "result": result,
                "summary": result.summary,
            },
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/outputs", response_class=HTMLResponse)
async def view_outputs(request: Request):
    """View past extraction outputs"""
    outputs = await _load_outputs()
    return templates.TemplateResponse("view_outputs.html", {"request": request, "outputs": outputs})

@app.get("/runs", response_class=HTMLResponse)
async def view_runs(request: Request, limit: int = 20):
    runs = await container.runlog_repo.get_recent(limit)
    return templates.TemplateResponse("view_runs.html", {"request": request, "runs": runs})

@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    r = await container.runlog_repo.get_by_id(run_id)
    if not r:
        raise HTTPException(404, "Run not found")

    # Access raw fields exactly as stored
    ef = getattr(r, "extracted_fields", {}) or {}
    filled_data = ef.get("filled_data", {})
    fields_html = _format_filled_data(filled_data)

    return templates.TemplateResponse("view_run.html", {"request": request, "run": r, "ef": ef, "fields_html": fields_html})

# --- Helpers ---
def _format_filled_data(data: Dict[str, Any], prefix: str = "") -> str:
    """Recursively format filled data for display"""
    html = ""
    for k, v in data.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            html += _format_filled_data(v, key)
        else:
            html += f'<div class="field"><span class="field-name">{key}:</span> <span class="field-value">{v}</span></div>'
    return html

async def _save_output(result: ExtractionResult):
    """Save result to MongoDB instead of a local file"""
    # Access the collection directly from the container's db instance
    collection = container.convo_repo.db.outputs 
    await collection.insert_one(result.model_dump())
    logger.info("Extraction result saved to Atlas.")

async def _load_outputs() -> List[Dict]:
    """Load outputs from MongoDB Atlas"""
    collection = container.convo_repo.db.outputs
    cursor = collection.find({}).sort("_id", -1)
    return await cursor.to_list(length=100)

@app.get("/forms/{form_id}/live", response_class=HTMLResponse)
async def live_extraction_page(request: Request, form_id: str):
    """Render the two-panel live extraction UI."""
    form = await container.form_repo.get_by_id(form_id)
    if not form:
        raise HTTPException(404, "Form not found")
    return templates.TemplateResponse(
        "live_extract.html",
        {"request": request, "form": form}
    )

@app.post("/api/live-extract")
async def api_live_extract(payload: LiveExtractRequest) -> JSONResponse:
    form = await container.form_repo.get_by_id(payload.form_id)
    if not form:
        raise HTTPException(404, "Form not found")

    context = payload.conversation.strip()
    if not context:
        raise HTTPException(400, "Conversation text is empty")

    requests: list[ExtractionRequest] = [
        ExtractionRequest(
            context=context,
            field_name=field_key,
            instruction=question
        )
        for field_key, question in form.fields.items()
    ]

    model = get_live_model()
    answers = await model.extract_batch(requests)

    result: Dict[str, str] = {}
    for req, answer in zip(requests, answers):
        if answer and answer.strip():
            result[req.field_name] = answer

    return JSONResponse(content=result)