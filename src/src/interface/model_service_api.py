import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..infrastructure.ai.local_model import FormStateModel, GemmaFunctionalModel
from ..infrastructure.ai.summarizer import LocalSummarizer, GemmaSummarizer, QwenSummarizer
from ..infrastructure.config import settings

logger = logging.getLogger(__name__)

model: Any = None
summarizer: Any = None


class ExtractRequest(BaseModel):
    input_str: str
    field_keys: list[str] = Field(default_factory=list)


class LiveExtractRequest(BaseModel):
    conversation_text: str
    form_name: str
    current_field_state: dict[str, Any] = Field(default_factory=dict)
    field_keys: list[str] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    text: str


def _build_runtime_models() -> tuple[Any, Any]:
    if settings.EXTRACTION_MODEL_TYPE == "form_state":
        extraction_model = FormStateModel(model_path=settings.FORM_STATE_MODEL_PATH)
    else:
        extraction_model = GemmaFunctionalModel(
            max_input_tokens=512,
            max_new_tokens=256,
            temperature=0.0,
            checkpoint_path="/app/data_generation/models/checkpoint-200",
        )

    if settings.SUMMARIZER_TYPE == "gemma":
        summary_model = GemmaSummarizer(model_path=settings.SUMMARIZER_MODEL_PATH)
    elif settings.SUMMARIZER_TYPE == "qwen":
        summary_model = QwenSummarizer(model_name=settings.SUMMARIZER_MODEL_PATH)
    else:
        summary_model = LocalSummarizer()

    return extraction_model, summary_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, summarizer
    model, summarizer = _build_runtime_models()
    logger.info("Model service started.")
    yield


app = FastAPI(title="ProductLabs Model Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract(request: ExtractRequest) -> dict[str, Any]:
    answers = await model.process_extraction_request(
        request.input_str,
        field_keys=request.field_keys or None,
    )
    return {"answers": answers}


@app.post("/live-extract")
async def live_extract(request: LiveExtractRequest) -> dict[str, Any]:
    answers = await model.process_live_update(
        conversation_text=request.conversation_text,
        form_name=request.form_name,
        current_field_state=request.current_field_state,
        field_keys=request.field_keys,
    )
    return {"answers": answers}


@app.post("/summarize")
async def summarize(request: SummarizeRequest) -> dict[str, str]:
    summary = await summarizer.summarize(request.text)
    return {"summary": summary}
