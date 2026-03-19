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

with open(datapoints_file, 'r') as f:
    datapoints = json.load(f)
with open(conversations_file, 'r') as f:
    conversations = json.load(f)

test_datapoints = datapoints[400:420]
conv_map = {c['conversation_id']: list(c['conversation'].items()) for c in conversations}

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
    import re
    import json
    clean_text = text.replace('\\"', '"')
    match = re.search(r'"[iI]nitial_?\s*fields"\s*:\s*(\{.*?\})', clean_text, re.DOTALL | re.IGNORECASE)
    if match:
        fields_str = match.group(1)
        try:
            return json.loads(fields_str)
        except Exception:
            try:
                fixed_str = re.sub(r',\s*}', '}', fields_str)
                return json.loads(fixed_str)
            except:
                pass
    return None

def get_diff(current, next_state):
    diff = {}
    if not next_state: return diff
    for k, v in next_state.items():
        curr_v = current.get(k)
        if str(curr_v).strip().lower() != str(v).strip().lower():
            diff[k] = v
    return diff

def run_verify():
    results = []
    
    for i, dp in enumerate(test_datapoints):
        conv_id = dp.get("_conversation_id")
        full_conv = conv_map.get(conv_id, [])
        new_line = dp['input']['new_line']
        
        current_idx = -1
        for idx, (speaker, text) in enumerate(full_conv):
            if f"{speaker}: {text}" == new_line:
                current_idx = idx
                break
        
        start_idx = max(0, current_idx - 3)
        context_lines = full_conv[start_idx:current_idx]
        context_str = "\\n".join([f"{s}: {t}" for s, t in context_lines])
        
        current_form_state = dp['input']['current_form_state']
        current_initial = current_form_state.get('Initial fields', {})
        if not current_initial and 'initial_fields' in current_form_state:
            current_initial = current_form_state['initial_fields']
            
        system_content = "You are a conversational form-filling assistant. Analyze the conversation and update the form state and summary accordingly."
        
        user_content = f"""Form: {dp['input']['form_name']}
Description: {dp['input']['form_description']}

Current Summary: {dp['input'].get('current_summary_state', 'N/A')}

Current Form State:
{json.dumps(current_form_state)}

Conversation Context:
{context_str}

New Line:
{new_line}"""

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

        predicted_initial = None
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
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            predicted_initial = parse_function_call(result_text)
            status = "Parsed"
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
        
        del inputs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        ideal_initial = dp['ideal_output'].get('next_form_state', {}).get('Initial fields', {})
        if not ideal_initial and 'initial_fields' in dp['ideal_output'].get('next_form_state', {}):
            ideal_initial = dp['ideal_output'].get('next_form_state', {})['initial_fields']
            
        expected_update = get_diff(current_initial, ideal_initial)
        predicted_update = get_diff(current_initial, predicted_initial) if predicted_initial else {}
        
        results.append({
            "sample_id": i + 1,
            "new_line": new_line,
            "status": status,
            "expected_update": expected_update,
            "predicted_update": predicted_update,
            "predicted_raw": predicted_initial
        })
        
        print(f"Processed Sample {i+1}")

    with open(os.path.join(base_path, "verification_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_verify()
