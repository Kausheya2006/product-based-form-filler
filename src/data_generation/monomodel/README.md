# MonoModel: Conversational Form-Filling Assistant

This folder contains a locally fine-tuned **Qwen2.5-1.5B-Instruct** model, specialized for conversational form-filling via tool calling.

The model acts as an assistant that reads a conversation and iteratively updates a structured JSON form state based on the newest information provided.

## Directory Structure
*   **`model/`**: Contains the fine-tuned LoRA adapter weights (`adapter_model.safetensors`) and configurations.
*   **`curate_data.py`**: Script used to prepare the raw dataset into the Qwen 2.5 instruction format.
*   **`train.py`**: The QLoRA fine-tuning script.
*   **`test_accuracy.py` / `manual_verify.py`**: Scripts for evaluating the model's accuracy on unseen data.

---

## How to Use the Model (Inference per Iteration)

The model is designed to process **one line of conversation at a time**, looking at the most recent context to decide if the form or summary needs updating.

### 1. Requirements

Make sure you have the necessary libraries installed in your Python environment:
```bash
pip install torch transformers peft bitsandbytes
```

### 2. Loading the Model

Since the model was trained using QLoRA (4-bit), you must load the base model (`Qwen/Qwen2.5-1.5B-Instruct`) in 4-bit and attach the fine-tuned adapter from the `model/` folder.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Define paths
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./model" # Path to this folder's 'model' directory

# 1. Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.padding_side = "left"

# 2. Setup 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 4. Attach fine-tuned adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

### 3. Formatting the Input

For each new line of conversation, you must provide the model with the current form state, the current summary, a short context window, and the new line.

```python
import json

# Define the tool schema
tools = [{
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
}]

# Define your current state
current_form_state = {
    "Initial fields": {
        "name": "John Doe",
        "age": "N/A"
    },
    "New fields": {}
}
current_summary = "A conversation with John Doe."
context_history = "Agent: What is your name?
John: I'm John Doe."
new_line = "Agent: And how old are you?
John: I am 35 years old."

# Construct the prompt
system_content = "You are a conversational form-filling assistant. Analyze the conversation and update the form state and summary accordingly."
user_content = f"""Form: Basic Info
Description: A basic information gathering form.

Current Summary: {current_summary}

Current Form State:
{json.dumps(current_form_state)}

Conversation Context:
{context_history}

New Line:
{new_line}"""

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]

# Apply Qwen's Chat Template
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to("cuda")
```

### 4. Generating the Output

```python
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256, # Adjust based on expected JSON size
        do_sample=False,
    )

# Decode only the newly generated tokens
new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
result_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(result_text)
```

### 5. Parsing the Result

The model will output an XML `<tool_call>` containing the JSON arguments. You can use a regex to extract it safely:

```python
import re

def parse_model_output(text):
    match = re.search(r'<tool_call>(.*?)(?:</tool_call>|$)', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except:
            # Fallback for minor formatting errors
            arg_match = re.search(r'"arguments":\s*"(.*?)"\}', json_str, re.DOTALL)
            if arg_match:
                args_str = arg_match.group(1).replace('"', '"').replace('
', '
')
                try:
                    return json.loads(args_str)
                except:
                    pass
    return None

parsed_json = parse_model_output(result_text)
if parsed_json:
    print("New Form State:", parsed_json.get("next_form_state"))
    print("New Summary:", parsed_json.get("new_summary_state"))
```

### Hardware Notes
* If running on a GPU with limited VRAM (e.g., 4GB GTX 1650), limit your `Conversation Context` to the **last 3-5 lines** to prevent `CUDA out of memory` errors during the forward pass.
* Always call `del inputs`, `del outputs`, `import gc; gc.collect()`, and `torch.cuda.empty_cache()` after each generation if you are processing samples in a loop.
