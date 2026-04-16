from uuid import uuid4
from datetime import datetime
import re
import asyncio 
from ..domain.domain import ExtractionResult, ExtractionRequest, RunLog
from ..domain.interfaces import IConversationRepository, IFormRepository, IExtractionModel, IPipeline, IRunLogRepository, ISummarizer
from ..domain.speakers import render_history_for_model

# Regex for basic email validation
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# Regex for phone numbers: optional +, digits, spaces, hyphens, parens  (7-15 digits)
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")

# Common date formats to try when parsing date strings
_DATE_FORMATS = [
    "%Y-%m-%d",       # 2024-03-07
    "%d-%m-%Y",       # 07-03-2024
    "%m/%d/%Y",       # 03/07/2024
    "%d/%m/%Y",       # 07/03/2024
    "%Y/%m/%d",       # 2024/03/07
    "%B %d, %Y",      # March 07, 2024
    "%b %d, %Y",      # Mar 07, 2024
    "%d %B %Y",       # 07 March 2024
    "%d %b %Y",       # 07 Mar 2024
    "%Y%m%d",         # 20240307
]


def _try_parse_date(value: str) -> str | None:
    """Try to parse *value* as a date using common formats. Returns ISO date string or None."""
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def validate_field_type(value, expected_type: str):
    """
    Validate that *value* conforms to *expected_type* (as stored in the form
    schema, e.g. "string", "int", "float", "email", "phone", "date").
    Returns the (possibly cast) value on success, or "N/A" when the value
    cannot be converted.
    """
    if value is None or (isinstance(value, str) and value.strip().upper() == "N/A"):
        return "N/A"

    expected_type = expected_type.strip().lower()

    if expected_type == "int":
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                return "N/A"
        return "N/A"

    if expected_type == "float":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return "N/A"
        return "N/A"

    if expected_type == "email":
        s = str(value).strip()
        if _EMAIL_RE.match(s):
            return s
        return "N/A"

    if expected_type == "phone":
        s = str(value).strip()
        # Check pattern and that there are at least 7 actual digits
        if _PHONE_RE.match(s) and len(re.sub(r"\D", "", s)) >= 7:
            return s
        return "N/A"

    if expected_type == "date":
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        s = str(value).strip()
        parsed = _try_parse_date(s)
        return parsed if parsed else "N/A"

    if expected_type == "string":
        return str(value)

    # Unknown / future types – accept as-is
    return value

class FormFillingService(IPipeline):
    def __init__(self, conversation_repo: IConversationRepository, form_repo: IFormRepository, extraction_model: IExtractionModel, runlog_repo: IRunLogRepository, summarizer: ISummarizer, model_type="field_process"): 
        self.convo_repo = conversation_repo
        self.form_repo = form_repo
        self.model = extraction_model
        self.runlog_repo = runlog_repo
        self.summarizer = summarizer 
        self.model_type = model_type

    @staticmethod
    def _flatten_dict(data: dict, prefix: str = "", out: dict | None = None) -> dict:
        if out is None:
            out = {}
        for key, value in (data or {}).items():
            dotted_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                FormFillingService._flatten_dict(value, dotted_key, out)
            else:
                out[dotted_key] = value
        return out

    @staticmethod
    def _answers_for_field_keys(raw_answers, field_keys: list[str], fallback_state: dict) -> list:
        if isinstance(raw_answers, dict):
            filled_data = raw_answers.get("filled_data", raw_answers)
            flat = (
                FormFillingService._flatten_dict(filled_data)
                if isinstance(filled_data, dict)
                else {}
            )
            return [flat.get(key, fallback_state.get(key, "N/A")) for key in field_keys]
        return list(raw_answers or [])

    async def run(self, conversation_id: str, form_id: str, version_index: int, owner_id: str = None) -> ExtractionResult:
        run_id = str(uuid4())

        await self.runlog_repo.create(RunLog(
            run_id=run_id,
            conversation_id=conversation_id,
            version_index=version_index,
            started_at=datetime.utcnow(),
            status="running",
            owner_id=owner_id,
        ))

        try:
            # 1. Fetch Data
            convo = await self.convo_repo.get_by_id(conversation_id)
            if not convo:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            form = await self.form_repo.get_by_id(form_id)
            if not form:
                raise ValueError(f"Form {form_id} not found")

            # 2. Prepare Context & Batch Requests
            context = convo.full_text
            requests = []
            field_keys = []
            empty_fields = {}
            full_convo = render_history_for_model(convo.latest_history)

            field_types = {}  # field_key -> expected type
            for field_key, field_type in form.fields.items():
                field_keys.append(field_key)
                field_types[field_key] = field_type
                if self.model_type == "field_process":
                    req = ExtractionRequest(
                        context=context,
                        field_name=field_key,
                        instruction=field_type,
                        original_type_hint=field_type
                    )
                    requests.append(req)
                elif self.model_type == "full_process":
                    empty_fields[field_key] = "N/A"
                else:
                    raise Exception

            summarization_task = asyncio.create_task(self.summarizer.summarize(context))

            if self.model_type == "field_process":
                extraction_task = self.model.extract_batch(requests) 
                answers, summary = await asyncio.gather(extraction_task, summarization_task)
            elif self.model_type == "full_process":
                if not hasattr(self.model, "process_live_update"):
                    raise RuntimeError(
                        "Static extraction requires a model with process_live_update "
                        "so it can use the live extraction prompt format."
                    )
                answers = [empty_fields[key] for key in field_keys]
                running_state = dict(empty_fields)
                lines = [line for line in full_convo.splitlines() if line.strip()]
                for index in range(len(lines)):
                    raw_answers = await self.model.process_live_update(
                        conversation_text="\n".join(lines[:index + 1]),
                        form_name=form.name,
                        current_field_state=running_state,
                        field_keys=field_keys,
                    )
                    answers = self._answers_for_field_keys(raw_answers, field_keys, running_state)
                    running_state = {
                        field_key: value
                        for field_key, value in zip(field_keys, answers)
                    }
                summary = await summarization_task
            else:
                raise Exception 

            # 4. Validate predicted types & compile results
            filled_data = {}
            for key, value in zip(field_keys, answers):
                expected = field_types.get(key, "string")
                value = validate_field_type(value, expected)

                if "." in key:
                    parts = key.split(".")
                    current = filled_data
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    filled_data[key] = value

            result = ExtractionResult(
                conversation_id=conversation_id,
                form_id=form_id,
                filled_data=filled_data,
                run_id=run_id,
                summary=summary  
            )

            await self.runlog_repo.update(run_id, {
                "finished_at": datetime.utcnow(),
                "status": "success",
                "summary": summary, 
                "extracted_fields": result.model_dump()
            })

            return result
        except Exception as e:
            await self.runlog_repo.update(run_id, {
                "finished_at": datetime.utcnow(),
                "status": "failed",
                "error": str(e)
            })
            raise e
