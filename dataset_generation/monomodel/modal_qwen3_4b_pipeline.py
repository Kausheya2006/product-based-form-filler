import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "monomodel-qwen3-4b"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
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
        "matplotlib>=3.9.0",
    )
)
outputs = modal.Volume.from_name("monomodel-qwen3-4b-outputs", create_if_missing=True)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "update_form_state",
            "description": "Update form state from newest conversation line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "string"},
                    "next_form_state": {"type": "object"},
                    "new_summary_state": {"type": "string"},
                },
                "required": ["thinking", "next_form_state", "new_summary_state"],
            },
        },
    }
]
SYSTEM_PROMPT = (
    "You are strict form-state updater.\n"
    "Use only NEW line plus short context.\n"
    "Rules:\n"
    "1) Keep old values unless NEW clearly changes them.\n"
    "2) Prefer editing existing known keys.\n"
    "3) Add New fields only when no known key fits concrete form-relevant fact.\n"
    "4) Explicit negation/cancel/unknown -> set impacted value to N/A.\n"
    "5) If NEW has no actionable form info, return empty updates.\n"
    "6) next_form_state must be object with keys: Initial fields, New fields.\n"
    "Return one function call to update_form_state. Keep thinking very short."
)
PLACEHOLDERS = {"", "n/a", "na", "none", "null", "unknown", "not available"}


def nrm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def nrm_value(value: Any) -> str:
    text = nrm_space(value)
    return "N/A" if text.lower() in PLACEHOLDERS else (text or "N/A")


def stringify_new_line(value: Any) -> str:
    if isinstance(value, str):
        return nrm_space(value)
    if isinstance(value, dict) and len(value) == 1:
        speaker, text = next(iter(value.items()))
        return f"{nrm_space(speaker)}: {nrm_space(text)}"
    return nrm_space(json.dumps(value, ensure_ascii=True))


def normalize_state(state: Any, initial_keys: list[str] | None = None) -> dict[str, dict[str, str]]:
    if not isinstance(state, dict):
        state = {}
    initial = state.get("Initial fields", state.get("initial_fields", {}))
    new_fields = state.get("New fields", state.get("new_fields", {}))
    if not isinstance(initial, dict):
        initial = {}
    if not isinstance(new_fields, dict):
        new_fields = {}

    initial_clean = {str(k).strip(): nrm_value(v) for k, v in initial.items() if str(k).strip()}
    if initial_keys:
        initial_clean = {k: initial_clean.get(k, "N/A") for k in initial_keys}

    new_clean = {}
    for k, v in new_fields.items():
        key = str(k).strip()
        if not key or key in initial_clean:
            continue
        val = nrm_value(v)
        if val != "N/A":
            new_clean[key] = val
    return {"Initial fields": initial_clean, "New fields": new_clean}


def merge_state(current_state: Any, update_state: Any) -> dict[str, dict[str, str]]:
    current = normalize_state(current_state)
    merged = normalize_state(update_state, initial_keys=list(current["Initial fields"].keys()))
    out = dict(current["Initial fields"])
    out.update(merged["Initial fields"])
    return {"Initial fields": out, "New fields": dict(merged["New fields"])}


def build_delta_state(current_state: Any, full_next_state: Any) -> dict[str, dict[str, str]]:
    current = normalize_state(current_state)
    full_next = merge_state(current, full_next_state)
    delta_initial = {}
    for k, v in full_next["Initial fields"].items():
        if current["Initial fields"].get(k, "N/A") != v:
            delta_initial[k] = v
    delta_new = {}
    cur_new = current["New fields"]
    nxt_new = full_next["New fields"]
    for k, v in nxt_new.items():
        if cur_new.get(k, "N/A") != v:
            delta_new[k] = v
    for k in cur_new:
        if k not in nxt_new:
            delta_new[k] = "N/A"
    return {"Initial fields": delta_initial, "New fields": delta_new}


def classify_op(current_state: Any, next_state: Any) -> str:
    delta = build_delta_state(current_state, next_state)
    if not delta["Initial fields"] and not delta["New fields"]:
        return "no_change"
    if any(v == "N/A" for v in delta["Initial fields"].values()) or any(v == "N/A" for v in delta["New fields"].values()):
        return "clear"
    if delta["New fields"]:
        return "add_new"
    cur = normalize_state(current_state)["Initial fields"]
    for key, val in delta["Initial fields"].items():
        if cur.get(key, "N/A") != "N/A":
            return "correct"
    return "fill"


def short_reason(op: str) -> str:
    return {
        "no_change": "No actionable form update.",
        "fill": "Fill newly provided values.",
        "correct": "Correct previously set values.",
        "clear": "Clear values from explicit negation.",
        "add_new": "Add concrete extra fields.",
    }.get(op, "Apply detected updates.")


def build_messages(
    form_name: str,
    form_description: str,
    current_summary: str,
    current_state: dict[str, dict[str, str]],
    context_lines: list[str],
    new_line: str,
) -> list[dict[str, str]]:
    state_obj = {
        "Known": list(current_state["Initial fields"].keys()),
        "Filled": {k: v for k, v in current_state["Initial fields"].items() if v != "N/A"},
    }
    if current_state["New fields"]:
        state_obj["New"] = current_state["New fields"]
    user = (
        f"F:{nrm_space(form_name)[:80]}\n"
        f"D:{nrm_space(form_description)[:140]}\n"
        f"S:{nrm_space(current_summary)[:220]}\n"
        f"STATE:{json.dumps(state_obj, ensure_ascii=True, separators=(',', ':'))}\n"
        f"CTX:{' || '.join(context_lines) if context_lines else '(none)'}\n"
        f"NEW:{new_line[:360]}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_tool_output(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    tool_match = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.DOTALL)
    candidate = tool_match.group(1).strip() if tool_match else text.strip()
    json_match = re.search(r"\{[\s\S]*\}", candidate)
    if json_match:
        candidate = json_match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("arguments", payload)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    next_state = args.get("next_form_state", {})
    summary = args.get("new_summary_state", "")
    if not isinstance(next_state, dict):
        return None
    if not isinstance(summary, str):
        summary = ""
    return {"next_form_state": next_state, "new_summary_state": nrm_space(summary)}


def context_from_input(before: Any) -> list[str]:
    if isinstance(before, dict):
        out = []
        for raw_speaker, text in list(before.items())[-3:]:
            speaker = str(raw_speaker).rsplit(" ", 1)[0]
            out.append(f"{speaker}: {text}"[:180])
        return out
    if isinstance(before, list):
        return [str(x)[:180] for x in before[-3:]]
    if isinstance(before, str) and before.strip():
        return [before.strip()[:180]]
    return []


@app.function(
    image=image,
    gpu="T4",
    cpu=0.125,
    memory=256,
    timeout=60 * 60,
    volumes={"/root/outputs": outputs},
)
def train_eval(datapoints: list[dict], max_steps: int = 80, epochs: float = 1.5, seed: int = 42) -> dict:
    import gc

    import matplotlib.pyplot as plt
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    adapter_repo_id = os.environ.get("HF_ADAPTER_REPO_ID", "").strip()
    out_dir = Path("/root/outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    curated = []
    skipped = 0
    for dp in datapoints:
        if not isinstance(dp, dict) or not isinstance(dp.get("input"), dict) or not isinstance(dp.get("ideal_output"), dict):
            skipped += 1
            continue
        conv_id = dp.get("_conversation_id")
        if not isinstance(conv_id, str):
            skipped += 1
            continue
        inp = dp["input"]
        ideal = dp["ideal_output"]
        new_line = stringify_new_line(inp.get("new_line", ""))
        context_lines = context_from_input(inp.get("10_lines_before"))
        current_state = normalize_state(inp.get("current_form_state", {}))
        full_next_state = merge_state(current_state, ideal.get("next_form_state", {}))
        delta_state = build_delta_state(current_state, full_next_state)
        op = classify_op(current_state, full_next_state)
        msgs = build_messages(
            form_name=str(inp.get("form_name", "Form")),
            form_description=str(inp.get("form_description", "")),
            current_summary=str(inp.get("current_summary_state", "")),
            current_state=current_state,
            context_lines=context_lines,
            new_line=new_line,
        )
        args = {
            "thinking": short_reason(op),
            "next_form_state": delta_state,
            "new_summary_state": str(ideal.get("new_summary_state", ""))[:220],
        }
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "update_form_state",
                            "arguments": json.dumps(args, ensure_ascii=True, separators=(",", ":")),
                        },
                    }
                ],
            }
        )
        text = tokenizer.apply_chat_template(
            msgs,
            tools=TOOL_SCHEMA,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        curated.append({"text": text, "conversation_id": conv_id, "operation_type": op, "raw": dp})

    grouped = defaultdict(list)
    for row in curated:
        grouped[row["conversation_id"]].append(row)
    convo_ids = list(grouped.keys())
    random.Random(seed).shuffle(convo_ids)
    test_n = max(1, round(0.2 * len(convo_ids)))
    val_n = max(1, round(0.2 * len(convo_ids)))
    test_ids = set(convo_ids[:test_n])
    val_ids = set(convo_ids[test_n:test_n + val_n])
    train_ids = set(convo_ids[test_n + val_n:])
    if not train_ids:
        train_ids = set(list(val_ids)[:1])
        val_ids = set(list(val_ids)[1:])

    train_rows = [r for cid in convo_ids if cid in train_ids for r in grouped[cid]]
    val_rows = [r for cid in convo_ids if cid in val_ids for r in grouped[cid]]
    test_rows = [r for cid in convo_ids if cid in test_ids for r in grouped[cid]]

    weights = {"clear": 8, "correct": 6, "add_new": 4, "fill": 3, "no_change": 1}
    train_text = []
    for row in train_rows:
        for _ in range(max(1, weights.get(row["operation_type"], 1))):
            train_text.append({"text": row["text"]})
    random.Random(seed + 1).shuffle(train_text)
    train_ds = Dataset.from_list(train_text)
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_rows])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_cfg = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        logging_steps=2,
        num_train_epochs=epochs,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_length=384,
        gradient_checkpointing=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        args=train_cfg,
    )
    trainer.train()

    model_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    pushed_adapter_repo = ""
    if adapter_repo_id:
        try:
            trainer.model.push_to_hub(adapter_repo_id)
            tokenizer.push_to_hub(adapter_repo_id)
            pushed_adapter_repo = adapter_repo_id
        except Exception:
            pushed_adapter_repo = ""

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for row in trainer.state.log_history:
        if "loss" in row and "step" in row:
            train_steps.append(row["step"])
            train_loss.append(row["loss"])
        if "eval_loss" in row and "step" in row:
            eval_steps.append(row["step"])
            eval_loss.append(row["eval_loss"])
    plt.figure(figsize=(7, 4))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Qwen3-4B LoRA training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=140)
    plt.close()

    eval_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token or eval_tokenizer.unk_token
    eval_tokenizer.padding_side = "left"
    base_eval = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    eval_model = PeftModel.from_pretrained(base_eval, str(model_dir))
    eval_model.eval()

    metrics = Counter()
    rows = []
    exp_ops = Counter()
    pred_ops = Counter()
    for idx, row in enumerate(test_rows, start=1):
        inp = row["raw"]["input"]
        ideal = row["raw"]["ideal_output"]
        conv_id = row["conversation_id"]
        new_line = stringify_new_line(inp.get("new_line", ""))
        context_lines = context_from_input(inp.get("10_lines_before"))
        current_state = normalize_state(inp.get("current_form_state", {}))
        expected_state = merge_state(current_state, ideal.get("next_form_state", {}))
        exp_op = classify_op(current_state, expected_state)
        exp_ops[exp_op] += 1
        messages = build_messages(
            form_name=str(inp.get("form_name", "Form")),
            form_description=str(inp.get("form_description", "")),
            current_summary=str(inp.get("current_summary_state", "")),
            current_state=current_state,
            context_lines=context_lines,
            new_line=new_line,
        )
        inputs = eval_tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMA,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to("cuda")
        with torch.no_grad():
            out = eval_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=224,
                do_sample=False,
                pad_token_id=eval_tokenizer.pad_token_id,
                eos_token_id=eval_tokenizer.eos_token_id,
            )
        text = eval_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_tool_output(text)
        if parsed is None:
            metrics["parse_fail"] += 1
            pred_state = current_state
        else:
            metrics["parse_ok"] += 1
            pred_state = merge_state(current_state, parsed.get("next_form_state", {}))
        pred_op = classify_op(current_state, pred_state)
        pred_ops[pred_op] += 1
        exp_changed = exp_op != "no_change"
        pred_changed = pred_op != "no_change"
        if exp_changed and pred_changed:
            metrics["change_tp"] += 1
        elif exp_changed and not pred_changed:
            metrics["change_fn"] += 1
        elif not exp_changed and pred_changed:
            metrics["change_fp"] += 1
        else:
            metrics["change_tn"] += 1
        if pred_state == expected_state:
            metrics["exact"] += 1
        metrics["total"] += 1
        rows.append(
            {
                "sample": idx,
                "conversation_id": conv_id,
                "expected_op": exp_op,
                "predicted_op": pred_op,
                "exact": pred_state == expected_state,
            }
        )
        del out
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    def ratio(a: int, b: int) -> float:
        return 0.0 if b <= 0 else a / b

    eval_report = {
        "metrics": dict(metrics),
        "exact_state_accuracy": round(100 * ratio(metrics["exact"], max(metrics["total"], 1)), 2),
        "change_precision": round(100 * ratio(metrics["change_tp"], metrics["change_tp"] + metrics["change_fp"]), 2),
        "change_recall": round(100 * ratio(metrics["change_tp"], metrics["change_tp"] + metrics["change_fn"]), 2),
        "expected_ops": dict(exp_ops),
        "predicted_ops": dict(pred_ops),
        "rows": rows,
    }
    split_report = {
        "small_datapoints_total": len(datapoints),
        "curated_rows": len(curated),
        "skipped_rows": skipped,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "operation_distribution_total": dict(Counter(r["operation_type"] for r in curated)),
    }
    train_report = {
        "model_id": MODEL_ID,
        "max_steps": max_steps,
        "epochs": epochs,
        "log_history": trainer.state.log_history,
    }
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(split_report, file, indent=2, ensure_ascii=True)
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(train_report, file, indent=2, ensure_ascii=True)
    with (out_dir / "eval_results.json").open("w", encoding="utf-8") as file:
        json.dump(eval_report, file, indent=2, ensure_ascii=True)

    outputs.commit()
    return {
        "run_id": run_id,
        "output_dir_in_volume": f"/root/outputs/{run_id}",
        "split_summary": split_report,
        "eval_summary": {
            "exact_state_accuracy": eval_report["exact_state_accuracy"],
            "change_precision": eval_report["change_precision"],
            "change_recall": eval_report["change_recall"],
            "parse_ok": eval_report["metrics"].get("parse_ok", 0),
            "parse_fail": eval_report["metrics"].get("parse_fail", 0),
        },
        "pushed_adapter_repo": pushed_adapter_repo,
    }


@app.local_entrypoint()
def main(max_steps: int = 80, epochs: float = 1.5):
    with open(LOCAL_SMALL_DP, "r", encoding="utf-8") as file:
        datapoints = json.load(file)
    result = train_eval.remote(datapoints=datapoints, max_steps=max_steps, epochs=epochs)
    print(json.dumps(result, indent=2))
