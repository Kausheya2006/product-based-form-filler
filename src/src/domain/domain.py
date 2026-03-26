from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from .speakers import render_history_for_model

class ConversationVersion(BaseModel):
    version_index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    history: Dict[str, str]
    run_id: str | None = None
    source_mode: str = "text"
    input_language: str | None = None
    raw_transcript: str | None = None
    translated_transcript: str | None = None

class Conversation(BaseModel):
    id: str = Field(alias="conversation_id")
    form_id: str = Field(alias="form_id")
    name: str = Field(default="", alias="conversation_name")
    versions: List[ConversationVersion] = [] 
    owner_id: Optional[str] = None

    @property
    def latest_history(self) -> Dict[str, str]:
        """Returns the most recent version of the conversation."""
        if not self.versions:
            return {}
        return sorted(self.versions, key=lambda x: x.version_index)[-1].history

    @property
    def full_text(self) -> str:
        """Combines all turns from the LATEST version into a single string."""
        return render_history_for_model(self.latest_history)

class FormSchema(BaseModel):
    id: str = Field(alias="form_id")
    name: str = Field(alias="form_name")
    description: str = ""
    owner_id: Optional[str] = None
    # Simplified schema: field_name -> type_description
    # We flattened the address in the JSONL to make parallel extraction easier
    fields: Dict[str, str] = Field(alias="schema")

class ExtractionResult(BaseModel):
    conversation_id: str
    form_id: str
    filled_data: Dict[str, Any]
    run_id: str
    summary: str

class ExtractionRequest(BaseModel):
    context: str
    field_name: str
    instruction: str
    original_type_hint: str = "string"

class RunLog(BaseModel):
    run_id: str
    conversation_id: str
    version_index: int
    started_at: datetime
    finished_at: datetime | None = None
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    status: str  # "running", "success", "failed"
    error: str | None = None
    summary: str | None = None
