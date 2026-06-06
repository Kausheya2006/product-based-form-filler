import json
import os
import re
from typing import Any

import modal

APP_NAME = "monomodel-qwen3-4b-infer"
MODEL_ID = os.environ.get("QWEN3_MODAL_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
LORA_ADAPTER_ID = os.environ.get("QWEN3_MODAL_LORA_ADAPTER_ID", "").strip()

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers>=4.56.0",
        "bitsandbytes>=0.43.3",
        "peft>=0.12.0",
    )
)

_MODEL = None
_TOKENIZER = None


def _nrm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _nrm_value(value: Any) -> str:
    text = _nrm_space(value)
    return text if text else "N/A"


def _build_prompt(
    form_name: str,
    current_field_state: dict[str, Any],
    conversation_text: str,
    accepted_new_fields: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    seeded = {k: _nrm_value(v) for k, v in (current_field_state or {}).items()}
    accepted = {k: _nrm_value(v) for k, v in (accepted_new_fields or {}).items() if str(k).strip()}
    system = (
        "You are strict form-state updater.\n"
        "Use latest conversation text.\n"
        "Rules:\n"
        "1) Fill known keys from evidence.\n"
        "2) Keep previous value if no new evidence.\n"
        "3) Put unknown as N/A.\n"
        "4) Suggest new fields only for concrete extra facts.\n"
        "Return only JSON object with keys filled_data and suggested_new_fields."
    )
    user = (
        f"FORM:{_nrm_space(form_name)}\n"
        f"CURRENT:{json.dumps(seeded, ensure_ascii=True)}\n"
        f"ACCEPTED_NEW:{json.dumps(accepted, ensure_ascii=True)}\n"
        f"CONVO:{conversation_text[-2500:]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_legacy_extract_input(input_str: str) -> tuple[str, str, dict[str, Any]]:
    conversation_text = ""
    form_name = "Form"
    seeded = {}
    if "Conversation:" in input_str:
        after = input_str.split("Conversation:", 1)[1]
        if "Form:" in after:
            conversation_text = after.split("Form:", 1)[0].strip()
        else:
            conversation_text = after.strip()
    if "Form:" in input_str:
        after = input_str.split("Form:", 1)[1]
        form_name = after.splitlines()[0].strip() or "Form"
    if "Fields:" in input_str:
        raw = input_str.split("Fields:", 1)[1].strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                seeded = obj
        except json.JSONDecodeError:
            seeded = {}
    return conversation_text, form_name, seeded


def _generate_state_update(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    model, tokenizer = _load()
    field_keys = payload.get("field_keys") or []
    current_state = payload.get("current_field_state", {}) or {}
    seeded = {k: _nrm_value(current_state.get(k, "N/A")) for k in field_keys}
    messages = _build_prompt(
        form_name=str(payload.get("form_name", "Form")),
        current_field_state=current_state,
        conversation_text=str(payload.get("conversation_text", "")),
        accepted_new_fields=payload.get("accepted_new_fields", {}) or {},
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=240,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = _parse_json(text)

    filled = parsed.get("filled_data", {})
    new_fields = parsed.get("suggested_new_fields", {})
    if not isinstance(filled, dict):
        filled = {}
    if not isinstance(new_fields, dict):
        new_fields = {}

    merged_filled = {k: _nrm_value(filled.get(k, seeded.get(k, "N/A"))) for k in field_keys}
    cleaned_new = {}
    for key, value in new_fields.items():
        k = str(key).strip()
        if not k or k in merged_filled:
            continue
        v = _nrm_space(value)
        if v:
            cleaned_new[k] = v
    return merged_filled, cleaned_new


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if LORA_ADAPTER_ID:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, LORA_ADAPTER_ID)
        except Exception:
            pass
    model.eval()
    _MODEL, _TOKENIZER = model, tokenizer
    return _MODEL, _TOKENIZER


@app.function(image=image, gpu="T4", cpu=0.125, memory=256, timeout=60 * 20)
def modal_live_extract(payload: dict[str, Any]) -> dict[str, Any]:
    # Backward compatibility: extract mode payload has input_str+field_keys
    if payload.get("mode") == "extract" or ("input_str" in payload and "conversation_text" not in payload):
        input_str = str(payload.get("input_str", ""))
        conversation_text, form_name, seeded = _parse_legacy_extract_input(input_str)
        field_keys = payload.get("field_keys") or list(seeded.keys())
        extract_payload = {
            "conversation_text": conversation_text,
            "form_name": form_name,
            "current_field_state": seeded,
            "field_keys": field_keys,
            "accepted_new_fields": {},
        }
        merged_filled, _ = _generate_state_update(extract_payload)
        return {"answers": [merged_filled.get(k, "N/A") for k in field_keys]}

    merged_filled, cleaned_new = _generate_state_update(payload)
    return {
        "result": {
            "filled_data": merged_filled,
            "suggested_new_fields": cleaned_new,
        }
    }


@app.function(image=image, gpu="T4", cpu=0.125, memory=256, timeout=60 * 10)
def modal_summarize(payload: dict[str, Any]) -> dict[str, str]:
    import torch

    text = _nrm_space(payload.get("text", ""))
    if not text:
        return {"summary": ""}
    model, tokenizer = _load()
    messages = [
        {
            "role": "system",
            "content": "Summarize conversation in one compact paragraph. Output summary text only.",
        },
        {"role": "user", "content": text[-2500:]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    summary = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return {"summary": summary}


@app.local_entrypoint()
def smoke():
    sample = modal_live_extract.remote(
        {
            "conversation_text": "Agent: Your name? User: John Doe.",
            "form_name": "Demo Form",
            "current_field_state": {"full_name": "N/A"},
            "field_keys": ["full_name"],
            "accepted_new_fields": {},
        }
    )
    print(json.dumps(sample, indent=2))
