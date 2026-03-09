import google.generativeai as genai
import os
import json
import time
import re
from dotenv import load_dotenv

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "../.env"))
DATA_DIR = os.path.join(BASE_DIR, "../data")
CONVERSATIONS_PATH = os.path.join(DATA_DIR, "generated_conversations.json")
FORMS_PATH = os.path.join(DATA_DIR, "generated_forms.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "generated_linewise_data.json")

# Prompt Template
LINWISE_PROMPT = """You are an expert AI data annotator.
Your task is to analyze a conversation line by line and track how form fields are filled, corrected, or changed.

### FORM CONTEXT
Form: {form_name}
Description: {form_description}
Initial Schema (all values start as "N/A"):
{schema_fields}

### CONVERSATION
{conversation}

### TASK
Process the conversation line by line. For each line, identify any form fields that are updated based on the information provided in that line.
- If a line provides new information for a field, record the new value.
- If a line corrects or changes previously provided information, record the updated value.
- If a line does not provide any information relevant to the form fields, record an empty object {{}}.
- Maintain the exact order of the lines. 
- The output MUST be a JSON array of the SAME LENGTH as the number of lines in the conversation.

### RESPONSE FORMAT
Output ONLY a valid JSON array where each element corresponds to a line in the conversation (in the same order). 
Each element must be an object containing the field(s) that changed in that line.
If no fields changed, use an empty object {{}}.

Example:
[
  {{"field1": "value1"}},
  {{}},
  {{"field2": "updated_value"}},
  {{}}
]

NO markdown (no ```json), NO preamble, NO postamble. The response must start with `[` and end with `]`.
"""

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def append_to_json_file(filepath, new_item):
    """Appends a single item to a JSON array file efficiently."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([new_item], f, indent=2)
        return

    try:
        # Read the entire file, append, and write back (safest for structured JSON)
        # For very large files, a more complex file-pointer approach would be needed.
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data.append(new_item)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error appending to file: {e}")

def parse_json_response(text, expected_len):
    """Robustly extracts and parses JSON array from the response."""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    
    # If it still doesn't start with [, try to find the first [ and last ]
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start:end+1]
    
    try:
        data = json.loads(text)
        if isinstance(data, list):
            if len(data) == expected_len:
                return data
            else:
                print(f"  Warning: Expected {expected_len} items, got {len(data)}.")
                return None
        return None
    except json.JSONDecodeError:
        return None

def build_pairs():
    """Builds API Key + Model combinations for failover."""
    keys = []
    for i in range(1, 11):
        key_name = "GEMINI_API_KEY" if i == 1 else f"GEMINI_API_KEY_{i}"
        val = os.getenv(key_name)
        if val:
            keys.append(val)
            
    models = [
        "models/gemini-2.0-flash", 
        "models/gemini-2.5-flash", 
        "models/gemini-2.5-flash-lite"
    ]
    return [(k, m) for k in keys for m in models]

def is_quota_error(err_str):
    return "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str

def sort_conversation(convo_dict):
    """Sorts conversation lines by the numerical suffix in the keys."""
    def get_timestamp(key):
        match = re.search(r"(\d+)$", key)
        return int(match.group(1)) if match else 0
    
    sorted_keys = sorted(convo_dict.keys(), key=get_timestamp)
    return [(k, convo_dict[k]) for k in sorted_keys]

def generate_linewise_data():
    conversations = load_json(CONVERSATIONS_PATH)
    forms_list = load_json(FORMS_PATH)
    
    # Map form_id to form object
    forms = {str(f["form_id"]): f for f in forms_list}
    
    # Track processed conversations
    existing_data = load_json(OUTPUT_PATH)
    processed_convos = {item["conversation_id"] for item in existing_data}

    print(f"Total conversations: {len(conversations)}. Already processed: {len(processed_convos)}")
    
    pairs = build_pairs()
    if not pairs:
        print("No API keys found. Set GEMINI_API_KEY in .env")
        return

    pair_idx = 0
    api_key, model_name = pairs[pair_idx]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    for convo_obj in conversations:
        convo_id = convo_obj.get("conversation_id")
        if convo_id in processed_convos:
            continue
            
        form_id = str(convo_obj.get("form_id"))
        form_data = forms.get(form_id)
        if not form_data:
            print(f"Form {form_id} not found for convo {convo_id}. Skipping.")
            continue
            
        convo_dict = convo_obj.get("conversation", {})
        if not convo_dict:
            versions = convo_obj.get("versions", [])
            if versions:
                convo_dict = versions[0].get("history", {})
        
        if not convo_dict:
            print(f"No conversation content found for {convo_id}. Skipping.")
            continue

        sorted_lines = sort_conversation(convo_dict)
        formatted_convo = ""
        for i, (speaker, text) in enumerate(sorted_lines):
            formatted_convo += f"Line {i+1}: {speaker}: {text}\n"

        schema_fields_str = json.dumps(form_data.get("schema", {}), indent=2)
        
        prompt = LINWISE_PROMPT.format(
            form_name=form_data.get("form_name", "Unknown Form"),
            form_description=form_data.get("description", "No description provided."),
            schema_fields=schema_fields_str,
            conversation=formatted_convo
        )

        success = False
        attempts = 0
        quota_swaps_this_convo = 0
        
        while not success:
            try:
                print(f"Processing convo {convo_id} (Lines: {len(sorted_lines)}, Attempt: {attempts+1}) using {model_name}...")
                response = model.generate_content(prompt)
                linewise_data = parse_json_response(response.text, len(sorted_lines))
                
                if linewise_data:
                    output_entry = {
                        "conversation_id": convo_id,
                        "form_id": form_id,
                        "linewise_data": linewise_data
                    }
                    append_to_json_file(OUTPUT_PATH, output_entry)
                    processed_convos.add(convo_id)
                    print(f"  ✓ Success: Processed {convo_id}")
                    success = True
                else:
                    print(f"  ✗ Failed to parse JSON or length mismatch. Retrying...")
                    attempts += 1
                    if attempts >= 3:
                        print(f"  Skipping {convo_id} after 3 failed parsing attempts.")
                        break
                    time.sleep(2)
            except Exception as e:
                err_str = str(e)
                if is_quota_error(err_str):
                    print(f"  Quota limit on {model_name}. Rotating...")
                    pair_idx = (pair_idx + 1) % len(pairs)
                    quota_swaps_this_convo += 1
                    
                    if quota_swaps_this_convo >= len(pairs):
                        print("  All keys hit quota recently. Waiting 60s before trying again...")
                        time.sleep(60)
                        quota_swaps_this_convo = 0
                    
                    api_key, model_name = pairs[pair_idx]
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    time.sleep(2)
                else:
                    print(f"  ✗ Error: {err_str}")
                    attempts += 1
                    if attempts >= 3:
                        break
                    time.sleep(5)
        
        time.sleep(1.0) # Small delay

if __name__ == "__main__":
    generate_linewise_data()
