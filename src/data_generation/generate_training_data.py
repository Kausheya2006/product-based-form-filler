import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

PROMPT = '''Generate a dataset of {count} realistic conversations and forms with multiple fields relating to the conversation to extract information from the conversation to fill the form.

Output a JSON list of objects. Each object must have these keys:
- id (integer, start from the next available ID)
- conversation_title (string)
- conversation (string with multiple lines each separated by a \\n. Each line starts with a <Speaker Name>: <Text>.)
- form_name (string)
- form_desription (string)
- fields (nested JSON object containing multiple fields as keys and corresponding output value as values. In some cases the value can be N/A if answer not present in the conversation context.)

The conversations should be diverse and realistic. Examples of scenarios:
- Doctor and Patient consultations
- Hotel/Restaurant reservations
- Banking inquiries
- Tech support calls
- Car/Equipment rentals
- Insurance claims
- Job interviews
- Customer service interactions
- Legal consultations
- Real estate inquiries

Make sure:
1. Each conversation is unique and realistic
2. Forms have 5-10 relevant fields
3. Some fields should have N/A values (missing information)
4. Conversations should be natural and varied in length
5. Include different types of data: names, dates, numbers, descriptions, etc.'''

def generate_batch(model, batch_size, start_id, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(PROMPT.format(count=batch_size))
            data = json.loads(response.text)
            for i, item in enumerate(data):
                item['id'] = start_id + i
            return data
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(60)
            elif attempt < max_retries - 1:
                time.sleep(10)
    raise Exception("Failed to generate batch")

def generate_dataset(target=200, filepath="data/training_data.json"):
    api_keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_2")]
    models = ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"]
    
    existing = json.load(open(filepath)) if os.path.exists(filepath) else []
    next_id = max([item['id'] for item in existing], default=0) + 1
    
    for api_key in api_keys:
        if len(existing) >= target:
            break
        genai.configure(api_key=api_key)
        for model_name in models:
            if len(existing) >= target:
                break
            model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json", "temperature": 1.2})
            while len(existing) < target:
                try:
                    batch = generate_batch(model, min(10, target - len(existing)), next_id)
                    existing.extend(batch)
                    next_id += len(batch)
                    with open(filepath, 'w') as f:
                        json.dump(existing, f, indent=4)
                    time.sleep(2)
                except:
                    break
    return existing

if __name__ == "__main__":
    generate_dataset()
