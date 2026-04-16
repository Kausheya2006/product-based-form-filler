import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "monomodel-gemma4e2b-finetune"
BASE_MODEL_ID = os.environ.get("GEMMA4_BASE_MODEL_ID", "google/gemma-4-E2B-it")
LOCAL_SMALL_DP = (
    "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src/data/generated_small_datapoints.json"
)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers>=4.56.0",
        "datasets>=2.20.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.3",
        "peft>=0.12.0",
        "trl>=0.10.1",
    )
)
outputs = modal.Volume.from_name("monomodel-gemma4e2b-outputs", create_if_missing=True)


def _nrm(v) -> str:
    return str(v or "").strip()


def _extract_context(dp: dict) -> tuple[str, dict, str, str, str]:
    inp = dp.get("input", {})
    current = inp.get("current_form_state", {})
    before = inp.get("10_lines_before", {})
    convo = ""
    if isinstance(before, dict):
        convo = "\n".join(f"{str(k).rsplit(' ',1)[0]}: {v}" for k, v in before.items())
    elif isinstance(before, list):
        convo = "\n".join(str(x) for x in before)
    elif isinstance(before, str):
        convo = before
    new_line = inp.get("new_line", "")
    form_name = _nrm(inp.get("form_name", "Form"))
    return form_name, current if isinstance(current, dict) else {}, convo, _nrm(new_line), _nrm(inp.get("current_summary_state", ""))


def _to_prompt(dp: dict) -> str:
    form_name, current, convo, new_line, _ = _extract_context(dp)
    ideal = dp.get("ideal_output", {})
    target = {
        "filled_data": ideal.get("next_form_state", {}),
        "new_summary_state": ideal.get("new_summary_state", ""),
    }
    return (
        "<start>\n"
        "System: Update form fields from conversation. Keep unchanged fields stable.\n"
        f"Form: {form_name}\n"
        f"Current: {json.dumps(current, ensure_ascii=True)}\n"
        f"Context: {convo[-2000:]}\n"
        f"New: {new_line}\n"
        f"Answer JSON: {json.dumps(target, ensure_ascii=True)}\n"
        "</end>"
    )


def _to_infer_prompt(dp: dict) -> str:
    form_name, current, convo, new_line, summary = _extract_context(dp)
    return (
        "<start>\n"
        "System: Update form fields from conversation. Keep unchanged fields stable.\n"
        f"Form: {form_name}\n"
        f"Current: {json.dumps(current, ensure_ascii=True)}\n"
        f"CurrentSummary: {summary}\n"
        f"Context: {convo[-2000:]}\n"
        f"New: {new_line}\n"
        "Answer JSON only with keys filled_data and new_summary_state.\n"
        "</end>"
    )


def _parse_json_obj(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    raw = text[start:end + 1]
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_state_map(state: Any) -> dict[str, str]:
    if not isinstance(state, dict):
        return {}
    if isinstance(state.get("Initial fields"), dict):
        base = state.get("Initial fields", {})
    elif isinstance(state.get("initial_fields"), dict):
        base = state.get("initial_fields", {})
    else:
        base = state
    out = {}
    for k, v in base.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = _nrm(v) or "N/A"
    return out


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=0.125,
    memory=256,
    timeout=60 * 60,
    volumes={"/root/outputs": outputs},
)
def train_gemma(datapoints: list[dict], max_steps: int = 20, epochs: float = 1.0, seed: int = 42) -> dict:
    if "gguf" in BASE_MODEL_ID.lower():
        raise ValueError(
            "GGUF checkpoints are inference artifacts. Fine-tune with a non-GGUF base model ID, then export GGUF."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    rows = []
    for dp in datapoints:
        if isinstance(dp, dict) and isinstance(dp.get("input"), dict) and isinstance(dp.get("ideal_output"), dict):
            rows.append(
                {
                    "text": _to_prompt(dp),
                    "infer_prompt": _to_infer_prompt(dp),
                    "target_next_state": dp.get("ideal_output", {}).get("next_form_state", {}),
                    "target_summary": _nrm(dp.get("ideal_output", {}).get("new_summary_state", "")),
                    "conversation_id": dp.get("_conversation_id", "unknown"),
                    "raw_dp": dp,
                }
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
    ids = list(grouped.keys())
    random.Random(seed).shuffle(ids)
    val_n = max(1, round(0.2 * len(ids)))
    val_ids = set(ids[:val_n])
    train_rows = [r for cid in ids if cid not in val_ids for r in grouped[cid]]
    val_rows = [r for cid in ids if cid in val_ids for r in grouped[cid]]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("/root/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=2,
        num_train_epochs=epochs,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataset_text_field="text",
        max_length=384,
        gradient_checkpointing=True,
        packing=False,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list([{"text": x["text"]} for x in train_rows]),
        eval_dataset=Dataset.from_list([{"text": x["text"]} for x in val_rows]),
        peft_config=peft_cfg,
        processing_class=tokenizer,
        args=cfg,
    )
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Full validation generation eval
    parse_ok = 0
    exact_state = 0
    summary_nonempty = 0
    changed_tp = 0
    changed_fp = 0
    changed_fn = 0
    changed_tn = 0
    total_field_matches = 0
    total_field_count = 0
    eval_rows = []
    sample_val = val_rows
    model.eval()
    for idx, row in enumerate(sample_val, start=1):
        prompt = row["infer_prompt"]
        target_state = _normalize_state_map(row.get("target_next_state", {}))
        target_summary = row.get("target_summary", "")
        _, current_state_raw, _, _, _ = _extract_context(row.get("raw_dp", {}))
        current_state = _normalize_state_map(current_state_raw)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = _parse_json_obj(text)
        if parsed:
            parse_ok += 1
        pred_state = _normalize_state_map(parsed.get("filled_data", {}))
        pred_summary = _nrm(parsed.get("new_summary_state", ""))
        keys = sorted(set(target_state.keys()) | set(current_state.keys()) | set(pred_state.keys()))
        if all((pred_state.get(k, current_state.get(k, "N/A")) == target_state.get(k, current_state.get(k, "N/A"))) for k in keys):
            exact_state += 1
        for k in keys:
            pred_v = pred_state.get(k, current_state.get(k, "N/A"))
            tgt_v = target_state.get(k, current_state.get(k, "N/A"))
            total_field_count += 1
            if pred_v == tgt_v:
                total_field_matches += 1
        exp_changed = any(target_state.get(k, current_state.get(k, "N/A")) != current_state.get(k, "N/A") for k in keys)
        pred_changed = any(pred_state.get(k, current_state.get(k, "N/A")) != current_state.get(k, "N/A") for k in keys)
        if exp_changed and pred_changed:
            changed_tp += 1
        elif exp_changed and not pred_changed:
            changed_fn += 1
        elif not exp_changed and pred_changed:
            changed_fp += 1
        else:
            changed_tn += 1
        if pred_summary:
            summary_nonempty += 1
        eval_rows.append(
            {
                "sample": idx,
                "conversation_id": row.get("conversation_id"),
                "parse_ok": bool(parsed),
                "exact_state": all(
                    pred_state.get(k, current_state.get(k, "N/A")) == target_state.get(k, current_state.get(k, "N/A"))
                    for k in keys
                ),
                "pred_changed": pred_changed,
                "expected_changed": exp_changed,
                "pred_summary_nonempty": bool(pred_summary),
                "target_summary_nonempty": bool(target_summary),
            }
        )

    change_precision = 100 * (changed_tp / max(changed_tp + changed_fp, 1))
    change_recall = 100 * (changed_tp / max(changed_tp + changed_fn, 1))
    field_accuracy = 100 * (total_field_matches / max(total_field_count, 1))
    gen_eval = {
        "eval_samples": len(sample_val),
        "parse_success_rate": round(100 * (parse_ok / max(len(sample_val), 1)), 2),
        "exact_state_accuracy": round(100 * (exact_state / max(len(sample_val), 1)), 2),
        "field_accuracy": round(field_accuracy, 2),
        "change_precision": round(change_precision, 2),
        "change_recall": round(change_recall, 2),
        "nonempty_summary_rate": round(100 * (summary_nonempty / max(len(sample_val), 1)), 2),
        "change_confusion": {
            "tp": changed_tp,
            "fp": changed_fp,
            "fn": changed_fn,
            "tn": changed_tn,
        },
        "rows": eval_rows,
    }

    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "base_model_id": BASE_MODEL_ID,
                "max_steps": max_steps,
                "epochs": epochs,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "log_history": trainer.state.log_history,
                "generation_eval": gen_eval,
            },
            file,
            indent=2,
            ensure_ascii=True,
        )

    outputs.commit()
    return {
        "run_id": run_id,
        "base_model_id": BASE_MODEL_ID,
        "output_dir_in_volume": f"/root/outputs/{run_id}",
        "adapter_dir_in_volume": f"/root/outputs/{run_id}/adapter",
        "generation_eval": gen_eval,
    }


@app.local_entrypoint()
def main(max_steps: int = 20, epochs: float = 1.0):
    with open(LOCAL_SMALL_DP, "r", encoding="utf-8") as file:
        datapoints = json.load(file)
    print(json.dumps(train_gemma.remote(datapoints=datapoints, max_steps=max_steps, epochs=epochs), indent=2))
