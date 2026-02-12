import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import json
import re
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

def format_sample(item):
    fields_empty = {k: "N/A" for k in item['fields'].keys()}
    inp = f"""Extract info from conversation to fill form.

Conversation: {item['conversation']}
Form: {item['form_name']}
Fields: {json.dumps(fields_empty)}"""
    out = json.dumps(item['fields'])
    return {"text": f"<start_of_turn>user\n{inp}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn>", "input": inp, "output": out}

def prepare_data(filepath):
    data = json.load(open(filepath))
    formatted = [format_sample(item) for item in data]
    train_split = int(0.8 * len(formatted))
    val_split = int(0.9 * len(formatted))
    return (
        Dataset.from_list(formatted[:train_split]),
        Dataset.from_list(formatted[train_split:val_split]),
        Dataset.from_list(formatted[val_split:]),
    )

def _normalize_value(value):
    return re.sub(r"\s+", " ", value.strip().strip('"')).lower()

def _extract_model_output(text):
    marker = "<start_of_turn>model"
    if marker in text:
        text = text.split(marker, 1)[1]
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else ""

def _extract_fields_from_text(text):
    json_blob = _extract_model_output(text)
    pattern = r'"([^"\\]+)"\s*:\s*("(?:\\.|[^"\\])*"|[-+]?\d+(?:\.\d+)?|true|false|null)'
    fields = {}
    for key, raw_value in re.findall(pattern, json_blob, flags=re.IGNORECASE):
        key = _normalize_value(key)
        value = raw_value.strip()
        if value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = value[1:-1]
        fields[key] = _normalize_value(str(value))
    return fields

def _regex_field_accuracy(pred_text, expected_text):
    expected = _extract_fields_from_text(expected_text)
    predicted = _extract_fields_from_text(pred_text)
    if not expected:
        return 0.0, 0.0
    matched = sum(1 for key, value in expected.items() if key in predicted and predicted[key] == value)
    field_acc = matched / len(expected)
    exact = 1.0 if predicted == expected else 0.0
    return field_acc, exact

def _preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)

def _safe_token_ids(token_ids, tokenizer, pad_token_id):
    ids = np.asarray(token_ids)
    if ids.ndim > 1:
        ids = ids.reshape(-1)
    if ids.dtype.kind not in {"i", "u"}:
        ids = ids.astype(np.float32, copy=False)
        ids = np.nan_to_num(ids, nan=float(pad_token_id), posinf=float(pad_token_id), neginf=float(pad_token_id))
        ids = ids.astype(np.int64, copy=False)
    else:
        ids = ids.astype(np.int64, copy=False)

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    if vocab_size <= 0:
        return np.full_like(ids, fill_value=pad_token_id, dtype=np.int64)

    return np.clip(ids, 0, vocab_size - 1)

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

        field_scores, exact_scores = [], []
        for pred_ids, label_ids in zip(predictions, labels):
            pred_ids = _safe_token_ids(pred_ids, tokenizer, pad_token_id)
            clean_label_ids = np.where(label_ids != -100, label_ids, pad_token_id)
            clean_label_ids = _safe_token_ids(clean_label_ids, tokenizer, pad_token_id)
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=False)
            label_text = tokenizer.decode(clean_label_ids, skip_special_tokens=False)
            field_acc, exact = _regex_field_accuracy(pred_text, label_text)
            field_scores.append(field_acc)
            exact_scores.append(exact)

        return {
            "regex_acc": float(np.mean(field_scores) * 100) if field_scores else 0.0,
            "regex_exact_acc": float(np.mean(exact_scores) * 100) if exact_scores else 0.0,
        }
    return compute_metrics

def train_model(model_name, train_ds, eval_ds, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_length=512,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.01,
        logging_steps=10,
        torch_empty_cache_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_steps=50,
        eval_strategy="steps",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_regex_acc",
        greater_is_better=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        packing=False,
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    train_metrics = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    val_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="val")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, {"train": train_metrics, "val": val_metrics}

def evaluate(model, tokenizer, test_ds):
    model.eval()
    field_scores, exact_scores = [], []
    
    for example in test_ds:
        inputs = tokenizer(
            f"<start_of_turn>user\n{example['input']}<end_of_turn>\n<start_of_turn>model\n",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        field_acc, exact = _regex_field_accuracy(generated, example["output"])
        field_scores.append(field_acc)
        exact_scores.append(exact)

    return {
        "regex_acc": float(np.mean(field_scores) * 100) if field_scores else 0.0,
        "regex_exact_acc": float(np.mean(exact_scores) * 100) if exact_scores else 0.0,
    }

if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_data("data/training_data.json")
    model, tokenizer, train_val_metrics = train_model("google/functiongemma-270m-it", train_ds, val_ds, "data_generation/models")
    test_metrics = evaluate(model, tokenizer, test_ds)
    results = {**train_val_metrics, "test": test_metrics}
    json.dump(results, open("data_generation/models/results.json", 'w'), indent=2)
