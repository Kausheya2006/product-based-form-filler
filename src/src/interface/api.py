"""FastAPI Interface - Routes only, DI handled by dependencies.py"""
import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import List, Dict, Any
from uuid import uuid4

from ..domain.domain import Conversation, FormSchema, ExtractionResult
from .dependencies import container, Container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="src/interface/templates")

async def seed_data():
    """Seed MongoDB from JSON files"""
    if os.path.exists("data/conversations.json"):
        with open("data/conversations.json", "r") as f:
            for data in json.load(f):
                data["conversation_id"] = str(data["conversation_id"])
                await container.convo_repo.save(Conversation(**data))
        logger.info("Conversations seeded.")
    if os.path.exists("data/forms.json"):
        with open("data/forms.json", "r") as f:
            for data in json.load(f):
                data["form_id"] = str(data["form_id"])
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

@app.post("/conversations/create", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    form_id: str = Form(...),
    conversation_id: str = Form(""), 
    conversation_text: str = Form(...)
):
    """Create a new conversation and redirect to extraction"""
    # Generate ID if not provided
    if not conversation_id.strip():
        conversation_id = str(uuid4())[:8]
    
    # Parse conversation text into dict format
    conversation_dict = _parse_conversation_text(conversation_text)
    
    if not conversation_dict:
        raise HTTPException(400, "Could not parse conversation. Use format: 'Doctor: text' or 'Patient: text'")
    
    # Create and save conversation
    convo = Conversation(
        conversation_id=conversation_id,
        form_id=form_id,
        conversation=conversation_dict
    )
    await container.convo_repo.save(convo)
    logger.info(f"Saved conversation {conversation_id}")
    
    # Redirect to extraction
    return RedirectResponse(url=f"/extract/{form_id}/{conversation_id}", status_code=303)


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
        result = await container.pipeline.run(convo_id, form_id)
        _save_output(result)
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
            },
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/outputs", response_class=HTMLResponse)
async def view_outputs(request: Request):
    """View past extraction outputs"""
    outputs = _load_outputs()
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

def _save_output(result: ExtractionResult):
    """Append result to output.json"""
    outputs = _load_outputs()
    outputs.append(result.model_dump())
    with open("data/output.json", "w") as f:
        json.dump(outputs, f, indent=2)

def _load_outputs() -> List[Dict]:
    """Load existing outputs"""
    if os.path.exists("data/output.json"):
        try:
            with open("data/output.json", "r") as f:
                return json.load(f)
        except:
            pass
    return []
