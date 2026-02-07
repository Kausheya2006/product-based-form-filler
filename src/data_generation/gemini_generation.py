import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the model with a specific configuration for JSON
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        # You can also enforce a specific schema here if needed
    }
)

# Prompt asking for a specific JSON structure
prompt = """
Generate a dataset of conversations and forms with multiple fields \
relating to the conversation to extract information from the cnversation to fill the form.
Generate 5 such data points. Output a JSON list of objects.
Each object must have these keys:
- id (integer)
- conversation_title (string)
- conversation (string with multiple lines each seperated by a '\n'. Each line starts with a <Speaker Name>: <Text>.)
- form_name (string)
- form_desription (string)
- fields (nested JSON object containing multiple fields as keys and corresponding output value as values. \
    In some cases the value can be N/A if answer not present in the conversation context.)
The conversations can be of many different types. For example between Doctor and Patient, Banker and Customer, \
Hotel Receptionist and Customer, etc. The form fields should contain information related to the scenario in hand. 
It is not necessary that all required information to fill these fields is present in the conversation.
"""

response = model.generate_content(prompt)

# Parse and use the data
data = json.loads(response.text)
with open("data/training_data.json", 'w') as f:
    json.dump(data, f, indent=4)