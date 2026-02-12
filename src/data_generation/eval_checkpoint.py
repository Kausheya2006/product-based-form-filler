import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(example):
    if "conversation" not in example:
        raise ValueError("Input JSON must contain 'conversation'.")
    if "form_name" not in example:
        raise ValueError("Input JSON must contain 'form_name'.")

    fields = example.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("'fields' must be a JSON object if provided.")

    fields_empty = {k: "N/A" for k in fields.keys()}
    return f"""Extract info from conversation to fill form.

Conversation: {example['conversation']}
Form: {example['form_name']}
Fields: {json.dumps(fields_empty)}"""


def parse_model_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def load_single_example(input_path):
    with open(input_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("Input JSON list must contain exactly one example.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a single object or a list with one object.")

    return payload


def run_inference(checkpoint_path, input_path, max_input_tokens, max_new_tokens, temperature):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    sample = load_single_example(input_path)
    prompt = build_prompt(sample)
    full_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    prediction = parse_model_json(generated)

    return {
        "checkpoint": checkpoint_path,
        "input_file": input_path,
        "prompt": prompt,
        "generated_text": generated,
        "prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one-form inference on a saved checkpoint.")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to JSON containing one object with at least 'conversation' and 'form_name'.",
    )
    parser.add_argument(
        "--checkpoint",
        default="src/data_generation/models/checkpoint-200",
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Maximum prompt length in tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save inference output JSON.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint}")
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON path does not exist: {args.input_json}")

    result = run_inference(
        checkpoint_path=args.checkpoint,
        input_path=args.input_json,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
