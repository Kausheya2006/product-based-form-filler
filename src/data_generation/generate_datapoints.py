import google.generativeai as genai
import os
import json
import time
import random
from dotenv import load_dotenv

load_dotenv()

# File paths based on the requested structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
CONVERSATIONS_PATH = os.path.join(DATA_DIR, "generated_conversations.json")
FORMS_PATH = os.path.join(DATA_DIR, "generated_forms.json")
DATAPOINTS_PATH = os.path.join(DATA_DIR, "generated_datapoints.json")

DATAPOINTS_PER_CONVO = 3

# The curated, foolproof base prompt template to generate structured and varied data points.
BASE_PROMPT = '''You are an expert AI data annotator and conversation analyst.
Your goal is to generate high-quality training data for a model that extracts form data and summarizes conversations line-by-line in real-time.

Here is the context:
Form Name: {form_name}
Form Description: {form_description}
Initial Form Fields (all start as "N/A"):
{schema_fields}

Here is a full conversation (JSON with speaker+timestamp as keys, spoken text as values). Note: Conversations can feature a single speaker (e.g., thinking aloud) or multiple speakers.
{conversation}

Task:
1. Carefully read the conversation.
2. {focus_instruction}
3. Pick EXACTLY {num_datapoints} distinct lines from the conversation that fit this focus as your "new_line" targets. Spread them out chronologically if possible.
4. For each chosen line, imagine an AI model has processed the conversation chronologically from the very first line up to the line JUST BEFORE your chosen line. Determine the `current_form_state` and `current_summary_state` at that exact moment.
5. Then, determine how processing the chosen `new_line` changes the state to become the `next_form_state` and `new_summary_state` (if at all, depending on the focus).
6. The form state MUST be split into "Initial fields" (the ones provided above) and "New fields".
   - "Initial fields": All fields provided above must be present. If a field's value hasn't been established yet up to that point, its value MUST be "N/A".
   - "New fields": Fields NOT in the initial schema but recommended based on the conversation (use RARELY and wisely, ~10-15% of the time. Only add if the `new_line` explicitly adds to the form's requirements but isn't covered by initial fields). If none, keep it empty `{{}}`.
7. `10_lines_before`: Extract up to 10 lines immediately preceding the `new_line` in chronological order. If fewer than 10 exist, include all available.

Output EXACTLY a JSON array containing {num_datapoints} objects. Each object MUST strictly follow this exact schema:

[
  {{
    "input": {{
      "new_line": "<speaker label>: <the exact chosen line text>",
      "10_lines_before": {{
         "<speaker label 1>": "<text 1>",
         "<speaker label 2>": "<text 2>"
      }},
      "form_name": "{form_name}",
      "form_description": "{form_description}",
      "current_form_state": {{
        "Initial fields": {{
           "<field_1>": "<value or N/A>",
           "<field_2>": "<value or N/A>"
        }},
        "New fields": {{}}
      }},
      "current_summary_state": "<Summary of all the important things in the conversation up to the line BEFORE new_line>"
    }},
    "ideal_output": {{
      "thinking": "<Brief reasoning buffer (1-3 sentences). Explain how new_line affects (or does not affect) the form state and summary. Keep it snappy.>",
      "next_form_state": {{
        "Initial fields": {{
           "<field_1>": "<value or N/A>",
           "<field_2>": "<new_value_if_updated or N/A>"
        }},
        "New fields": {{}}
      }},
      "new_summary_state": "<Updated summary incorporating new_line context, OR identical to current_summary_state if new_line adds no value>"
    }}
  }}
]

Important: Ensure the JSON array contains EXACTLY {num_datapoints} objects by repeating the structure above. Do not output any markdown formatting around the JSON, no explanations, no wrappers. Just the raw, valid JSON array.
'''

FOCUS_INSTRUCTIONS = [
    {
        "weight": 0.50,
        "text": "FOCUS: Standard Information Extraction. Pick EXACTLY {num_datapoints} distinct lines where a speaker provides clear, new information that directly answers an empty form field or significantly adds a relevant detail to the summary."
    },
    {
        "weight": 0.25,
        "text": "FOCUS: Revisions & Corrections. Pick EXACTLY {num_datapoints} distinct lines where a speaker corrects previous information, changes their mind, says 'wait no', or scraps a previous answer. The form state should show a field changing from an old value to a new value or reverting back to 'N/A'. If no such corrections exist in this conversation, pick lines where information is hesitant or partially unclear."
    },
    {
        "weight": 0.25,
        "text": "FOCUS: No-Change / Filler. Pick EXACTLY {num_datapoints} distinct lines that are purely conversational filler, greetings, simple agreements ('yes', 'okay', 'right'), or off-topic tangents. For these specific lines, the `next_form_state` and `new_summary_state` MUST remain EXACTLY identical to the `current_form_state` and `current_summary_state` because the line adds absolutely no relevant information."
    }
]

def choose_focus_instruction(num_datapoints):
    """Probabilistically chooses a focus instruction based on weights."""
    choices = [item["text"].format(num_datapoints=num_datapoints) for item in FOCUS_INSTRUCTIONS]
    weights = [item["weight"] for item in FOCUS_INSTRUCTIONS]
    return random.choices(choices, weights=weights, k=1)[0]

def parse_json_response(text):
    """Safely extracts and parses JSON array from the response."""
    text = text.strip()
    if "```" in text:
        for start_ch, end_ch in [('[', ']'), ('{', '}')]:
            if start_ch in text:
                s = text.index(start_ch)
                e = text.rindex(end_ch) + 1
                text = text[s:e]
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Failed to decode JSON from response.")
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
    """Appends new items to a JSON array file without overwriting the entire file contents initially."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)
            
    # Read existing data
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            
    # Append new datapoints
    data.extend(new_data_list)
    
    # Write back the updated list safely
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def generate_datapoints():
    conversations = load_json(CONVERSATIONS_PATH)
    forms_list = load_json(FORMS_PATH)
    
    # Map form_id to form object for O(1) lookups
    forms = {str(f["form_id"]): f for f in forms_list}
    
    existing_datapoints = load_json(DATAPOINTS_PATH)
    
    # Track which conversations have already been processed to avoid duplicates on re-runs
    processed_convos = set()
    for dp in existing_datapoints:
        if "_conversation_id" in dp:
            processed_convos.add(dp["_conversation_id"])

    total_convos = len(conversations)
    print(f"Total conversations: {total_convos}. Already processed: {len(processed_convos)}")
    
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
            # Fallback if conversation is nested in versions
            versions = convo_obj.get("versions", [])
            if versions:
                convo_dict = versions[0].get("history", {})
                
        if len(convo_dict) < DATAPOINTS_PER_CONVO:
            print(f"Conversation {convo_id} has fewer than {DATAPOINTS_PER_CONVO} lines. Skipping.")
            continue

        schema_fields_str = json.dumps(form_data.get("schema", {}), indent=2)
        convo_str = json.dumps(convo_dict, indent=2)
        
        # Probabilistically select the focus instruction for this API call
        focus_instruction = choose_focus_instruction(DATAPOINTS_PER_CONVO)
        
        prompt = BASE_PROMPT.format(
            form_name=form_data.get("form_name", "Unknown Form"),
            form_description=form_data.get("description", "No description provided."),
            schema_fields=schema_fields_str,
            conversation=convo_str,
            focus_instruction=focus_instruction,
            num_datapoints=DATAPOINTS_PER_CONVO
        )
        
        success = False
        attempts = 0
        quota_swaps = 0
        while not success:
            try:
                print(f"Processing convo {convo_id} (Attempt {attempts + 1}) with focus: {focus_instruction[:30]}...")
                response = model.generate_content(prompt)
                datapoints = parse_json_response(response.text)
                
                # Validation: Check if it's a list and has exact required length
                if isinstance(datapoints, list) and len(datapoints) == DATAPOINTS_PER_CONVO:
                    # Inject conversation_id into the datapoints for future tracking
                    for dp in datapoints:
                        dp["_conversation_id"] = convo_id
                        
                    append_to_json_file(DATAPOINTS_PATH, datapoints)
                    processed_convos.add(convo_id)
                    print(f"  ✓ Success: Generated {len(datapoints)} datapoints for convo {convo_id}.")
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
        
        # Small delay to prevent hitting rate limits rapidly
        time.sleep(1.5)

if __name__ == "__main__":
    generate_datapoints()
