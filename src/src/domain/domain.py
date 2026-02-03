from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime

class Conversation(BaseModel):
    id: str = Field(alias="conversation_id")
    form_id: str = Field(alias="form_id")
    history: Dict[str, str] = Field(alias="conversation")

    @property
    def full_text(self) -> str:
        """Combines all turns into a single string for valid context."""
        return "\n".join([f"{k}: {v}" for k, v in self.history.items()])

class FormSchema(BaseModel):
    id: str = Field(alias="form_id")
    name: str = Field(alias="form_name")
    description: str = ""
    # Simplified schema: field_name -> type_description
    # We flattened the address in the JSONL to make parallel extraction easier
    fields: Dict[str, str] = Field(alias="schema")

class ExtractionResult(BaseModel):
    conversation_id: str
    form_id: str
    filled_data: Dict[str, Any]

class ExtractionRequest(BaseModel):
    context: str
    field_name: str
    instruction: str
    original_type_hint: str = "string"

class RunLog(BaseModel):
    run_id: str
    conversation_id: str
    started_at: datetime
    finished_at: datetime | None = None
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    status: str  # "running", "success", "failed"
    error: str | None = None
