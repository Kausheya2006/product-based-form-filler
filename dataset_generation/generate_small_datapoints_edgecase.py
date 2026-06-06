import google.generativeai as genai
import os
import json
import time
import random
import re
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
CONVERSATIONS_PATH = os.path.join(DATA_DIR, "generated_small_conversations_edgecase.json")
FORMS_PATH = os.path.join(DATA_DIR, "generated_forms.json")
DATAPOINTS_PATH = os.path.join(DATA_DIR, "generated_small_datapoints.json")

DATAPOINTS_PER_CONVO = 5

BASE_PROMPT = '''You are an expert AI data annotator specializing in edge cases. 
Your task is to generate EXACTLY {num_datapoints} training datapoints in a STRICT JSON ARRAY format based on the provided conversation and form.

### CONTEXT
Form: {form_name}
Description: {form_description}
Initial Schema (all values start as "N/A"):
{schema_fields}

### CONVERSATION (SHORT - max 10 words per line) - EDGE CASE FOCUSED
{conversation}

### TASK
1. Analyze the conversation. THIS IS AN EDGE CASE CONVERSATION — focus ONLY on lines with:
   - Corrections or revisions (user changes previously said information)
   - Deletions or retractions (user removes or undoes information)
   - Conflicts or clarifications (resolving contradictions)
   - State changes (NOT simple insertions)

2. Pick EXACTLY {num_datapoints} target lines that represent these edge cases.
3. For each target line:
   - Identify the `current_form_state` and `current_summary_state` BEFORE this line.
   - Extract the `10_lines_before` (if available).
   - Determine the `next_form_state` and `new_summary_state` AFTER this line is processed.
   - Use "thinking" to explain the logic briefly.
   - Split form states into "Initial fields" (all schema fields must exist) and "New fields" (rarely used).

### RESPONSE FORMAT
Output ONLY a valid JSON array of EXACTLY {num_datapoints} objects. 
NO markdown (no ```json), NO preamble, NO postamble. 
The response must start with `[` and end with `]`.

### OBJECT SCHEMA
{{
  "input": {{
    "new_line": "Speaker: Text",
    "10_lines_before": {{"Speaker": "Text", ...}},
    "form_name": "{form_name}",
    "form_description": "{form_description}",
    "current_form_state": {{"Initial fields": {{...}}, "New fields": {{}}}},
    "current_summary_state": "..."
  }},
  "ideal_output": {{
    "thinking": "...",
    "next_form_state": {{"Initial fields": {{...}}, "New fields": {{}}}},
    "new_summary_state": "..."
  }}
}}
'''

def parse_json_response(text):
    """Robustly extracts and parses JSON array from the response."""
    text = text.strip()
    
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start:end+1]
    
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return None
    except json.JSONDecodeError:
        try:
            objs = re.findall(r"\{[\s\S]*?\}", text)
            parsed_objs = []
            for obj in objs:
                try:
                    parsed_objs.append(json.loads(obj))
                except:
                    continue
            if parsed_objs:
                return parsed_objs
        except:
            pass
        return None

def build_pairs():
    """Builds API Key + Model combinations for failover, supporting up to 10 keys."""
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

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def append_to_json_file(filepath, new_data_list):
    """Appends new items to a JSON array file."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)
            
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            
    data.extend(new_data_list)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def generate_edge_case_datapoints():
    conversations = load_json(CONVERSATIONS_PATH)
    forms_list = load_json(FORMS_PATH)
    
    forms = {str(f["form_id"]): f for f in forms_list}
    
    existing_datapoints = load_json(DATAPOINTS_PATH)
    
    processed_convos = set()
    for dp in existing_datapoints:
        if "_conversation_id" in dp:
            processed_convos.add(dp["_conversation_id"])

    total_convos = len(conversations)
    print(f"Total edge case conversations: {total_convos}. Already processed: {len(processed_convos)}")
    print(f"Current datapoints count: {len(existing_datapoints)}. Will append new edge case datapoints.")
    
    pairs = build_pairs()
    if not pairs:
        print("No API keys/models configured. Please set GEMINI_API_KEY in .env")
        return

    api_key, model_name = pairs[0]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.7})
    pair_idx = 0

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
                
        if len(convo_dict) < DATAPOINTS_PER_CONVO:
            print(f"Conversation {convo_id} has fewer than {DATAPOINTS_PER_CONVO} lines. Skipping.")
            continue

        schema_fields_str = json.dumps(form_data.get("schema", {}), indent=2)
        convo_str = json.dumps(convo_dict, indent=2)
        
        prompt = BASE_PROMPT.format(
            form_name=form_data.get("form_name", "Unknown Form"),
            form_description=form_data.get("description", "No description provided."),
            schema_fields=schema_fields_str,
            conversation=convo_str,
            num_datapoints=DATAPOINTS_PER_CONVO
        )
        
        success = False
        attempts = 0
        quota_swaps = 0
        while not success:
            try:
                print(f"Processing edge case convo {convo_id} (Attempt {attempts + 1})...")
                response = model.generate_content(prompt)
                datapoints = parse_json_response(response.text)
                
                if isinstance(datapoints, list) and len(datapoints) == DATAPOINTS_PER_CONVO:
                    for dp in datapoints:
                        dp["_conversation_id"] = convo_id
                        
                    append_to_json_file(DATAPOINTS_PATH, datapoints)
                    processed_convos.add(convo_id)
                    current_total = len(load_json(DATAPOINTS_PATH))
                    print(f"  ✓ Success: Generated {len(datapoints)} edge case datapoints for convo {convo_id}. Total datapoints: {current_total}")
                    success = True
                else:
                    print(f"  ✗ Failure: Output shape invalid (Expected {DATAPOINTS_PER_CONVO} items, got {len(datapoints) if isinstance(datapoints, list) else type(datapoints)}). Retrying...")
                    attempts += 1
                    time.sleep(2)
            except Exception as e:
                err_str = str(e)
                if is_quota_error(err_str):
                    print(f"  Quota limit reached on {model_name}. Switching API Key/Model pair...")
                    pair_idx = (pair_idx + 1) % len(pairs)
                    quota_swaps += 1
                    if quota_swaps >= len(pairs):
                        print("All API key-model pairs exhausted. Exiting program.")
                        return
                    api_key, model_name = pairs[pair_idx]
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.7})
                else:
                    print(f"  ✗ Error during generation: {err_str}")
                    attempts += 1
                    time.sleep(2)
        
        time.sleep(1.5)

if __name__ == "__main__":
    generate_edge_case_datapoints()
