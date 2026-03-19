import json
import os
from transformers import AutoTokenizer

# Paths
base_path = "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src"
data_path = os.path.join(base_path, "data")
model_path = os.path.join(base_path, "data_generation/monomodel/base_model")
output_file = os.path.join(base_path, "data_generation/monomodel/curated_datapoints.json")
datapoints_file = os.path.join(data_path, "generated_datapoints.json")
conversations_file = os.path.join(data_path, "generated_conversations.json")

# 1. Define the tool
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

def load_data():
    with open(datapoints_file, 'r') as f:
        datapoints = json.load(f)
    with open(conversations_file, 'r') as f:
        conversations = json.load(f)
    return datapoints, conversations

def process_data():
    datapoints, conversations = load_data()
    
    # Map conversations by ID for quick access
    conv_map = {c['conversation_id']: list(c['conversation'].items()) for c in conversations}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    curated_data = []
    
    for dp in datapoints:
        conv_id = dp.get("_conversation_id")
        if not conv_id or conv_id not in conv_map:
            continue
        
        full_conv = conv_map[conv_id]
        new_line = dp['input']['new_line']
        
        if not isinstance(new_line, str):
            new_line = str(new_line)

        # Find the index of the current line in the full conversation
        current_idx = -1
        for i, (speaker, text) in enumerate(full_conv):
            if not isinstance(text, str):
                text = str(text)
            line_str = f"{speaker}: {text}"
            if line_str == new_line:
                current_idx = i
                break
        
        if current_idx == -1:
            for i, (speaker, text) in enumerate(full_conv):
                if not isinstance(text, str):
                    text = str(text)
                if text in new_line or (isinstance(new_line, str) and new_line in text):
                    current_idx = i
                    break
        
        # Get up to 10 lines before
        start_idx = max(0, current_idx - 10)
        context_lines = full_conv[start_idx:current_idx]
        context_str = "\n".join([f"{s}: {t}" for s, t in context_lines])
        
        # Build messages for Qwen
        system_content = "You are a conversational form-filling assistant. Analyze the conversation and update the form state and summary accordingly."
        
        user_content = (
            f"Form: {dp['input']['form_name']}\n"
            f"Description: {dp['input']['form_description']}\n\n"
            f"Current Summary: {dp['input'].get('current_summary_state', 'N/A')}\n\n"
            f"Current Form State:\n{json.dumps(dp['input']['current_form_state'])}\n\n"
            f"Conversation Context:\n{context_str}\n\n"
            f"New Line:\n{new_line}"
        )
        
        ideal_out = dp.get('ideal_output', {})
        thinking = ideal_out.get('thinking', 'Updating form state based on the latest information provided.')
        next_form_state = ideal_out.get('next_form_state', {})
        new_summary_state = ideal_out.get('new_summary_state', "")
        
        # Ensure arguments are valid
        arguments_dict = {
            "thinking": thinking,
            "next_form_state": next_form_state,
            "new_summary_state": new_summary_state
        }
        
        # Try keeping as dict first, HF apply_chat_template handles dicts or JSON strings
        # We will use JSON string here since it's more stable across tokenizers
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {
                "role": "assistant", 
                "tool_calls": [
                    {
                        "type": "function", 
                        "function": {
                            "name": "update_form_state", 
                            "arguments": json.dumps(arguments_dict)
                        }
                    }
                ]
            }
        ]
        
        # Apply the template
        try:
            training_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False
            )
            
            curated_data.append({
                "text": training_text,
                "messages": messages,
                "conversation_id": conv_id
            })
        except Exception as e:
            try:
                # Fallback to dict if JSON string fails
                messages[2]["tool_calls"][0]["function"]["arguments"] = arguments_dict
                training_text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False
                )
                curated_data.append({
                    "text": training_text,
                    "messages": messages,
                    "conversation_id": conv_id
                })
            except Exception as e2:
                print(f"Error applying template for {conv_id}: {e2}")

    # Save curated data
    with open(output_file, 'w') as f:
        json.dump(curated_data, f, indent=2)
    
    print(f"Successfully processed {len(curated_data)} datapoints.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    process_data()
