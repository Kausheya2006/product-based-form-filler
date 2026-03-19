import json
import os
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Paths
base_path = "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src/data_generation/monomodel"
model_id = os.path.join(base_path, "base_model")
adapter_path = os.path.join(base_path, "model")
datapoints_file = "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src/data/generated_datapoints.json"
conversations_file = "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src/data/generated_conversations.json"

# 1. Load Data
with open(datapoints_file, 'r') as f:
    datapoints = json.load(f)
with open(conversations_file, 'r') as f:
    conversations = json.load(f)

# Use indices 400 to 500 for test
test_datapoints = datapoints[400:500]
conv_map = {c['conversation_id']: list(c['conversation'].items()) for c in conversations}

# 2. Setup Tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "update_form_state",
            "description": "Updates the form state and summary based on the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "string", "description": "Explain the logic before updating."},
                    "next_form_state": {"type": "object", "description": "The updated form fields."},
                    "new_summary_state": {"type": "string", "description": "The updated summary."}
                },
                "required": ["thinking", "next_form_state", "new_summary_state"]
            }
        }
    }
]

# 3. Load Model and Tokenizer
print("Loading model and adapter...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Model loaded.")

def parse_function_call(text):
    """
    Highly resilient parser that uses regex to extract just the 'Initial fields' or 'initial_fields' object.
    """
    import re
    import json
    
    # Clean up escaped quotes that might be inside a stringified JSON
    clean_text = text.replace('\\"', '"')
    
    # Look for "Initial fields": {...} or "initial_fields": {...}
    # We use a non-greedy match to find the first closing brace that makes sense,
    # but since fields don't usually have nested braces, [^{}]* is safe for the inner content.
    match = re.search(r'"[iI]nitial_?\s*fields"\s*:\s*(\{.*?\})', clean_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        fields_str = match.group(1)
        try:
            # Attempt to parse just this subset
            fields_dict = json.loads(fields_str)
            return {"next_form_state": {"Initial fields": fields_dict}}
        except Exception as e:
            # If standard JSON parsing fails, try fixing missing quotes or trailing commas
            try:
                fixed_str = re.sub(r',\s*}', '}', fields_str)
                fields_dict = json.loads(fixed_str)
                return {"next_form_state": {"Initial fields": fields_dict}}
            except:
                pass
                
    return None

def run_test():
    score = 0
    total = len(test_datapoints)
    
    for i, dp in enumerate(test_datapoints):
        conv_id = dp.get("_conversation_id")
        full_conv = conv_map.get(conv_id, [])
        new_line = dp['input']['new_line']
        
        current_idx = -1
        for idx, (speaker, text) in enumerate(full_conv):
            if f"{speaker}: {text}" == new_line:
                current_idx = idx
                break
        
        # Truncate context string directly to save VRAM, ensuring system prompt is untouched
        start_idx = max(0, current_idx - 3) # Keep only last 3 lines to severely limit VRAM usage on GTX 1650
        context_lines = full_conv[start_idx:current_idx]
        context_str = "\n".join([f"{s}: {t}" for s, t in context_lines])
        
        system_content = "You are a conversational form-filling assistant. Analyze the conversation and update the form state and summary accordingly."
        user_content = (
            f"Form: {dp['input']['form_name']}\n"
            f"Description: {dp['input']['form_description']}\n\n"
            f"Current Summary: {dp['input'].get('current_summary_state', 'N/A')}\n\n"
            f"Current Form State:\n{json.dumps(dp['input']['current_form_state'])}\n\n"
            f"Conversation Context:\n{context_str}\n\n"
            f"New Line:\n{new_line}"
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
        
        prompt_len = inputs['input_ids'].shape[1]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        except torch.cuda.OutOfMemoryError:
            print(f"Sample {i+1}: SKIPPED (OOM Error on {prompt_len} tokens)")
            del inputs
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        parsed_output = parse_function_call(result_text)
        
        if parsed_output:
            pred_next_state = parsed_output.get('next_form_state', {})
            pred_initial = pred_next_state.get('Initial fields', {})
            if not pred_initial and 'initial_fields' in pred_next_state: # sometimes the model outputs snake_case instead
                pred_initial = pred_next_state['initial_fields']
            
            ideal_initial = dp['ideal_output'].get('next_form_state', {}).get('Initial fields', {})
            if not ideal_initial and 'initial_fields' in dp['ideal_output'].get('next_form_state', {}):
                ideal_initial = dp['ideal_output'].get('next_form_state', {})['initial_fields']
            
            match = True
            for key, expected_val in ideal_initial.items():
                pred_val = pred_initial.get(key)
                if str(pred_val).strip().lower() != str(expected_val).strip().lower():
                    match = False
                    break
            
            if match:
                score += 1
                print(f"Sample {i+1}: MATCH")
            else:
                print(f"Sample {i+1}: FAIL (Mismatch)")
        else:
            print(f"Sample {i+1}: FAIL (No parse)")
            if i < 3: # Print first 3 fails for debugging
                 print(f"--- FAILED RAW OUTPUT ---\n{result_text}\n-------------------------")
        
        # Free up VRAM after each iteration to prevent OOM on 4GB GPU
        del outputs
        del inputs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
            
    print(f"\nAccuracy: {score}/{total}")

if __name__ == "__main__":
    run_test()
