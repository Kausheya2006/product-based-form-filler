import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

FORM_TYPES = [
    "Census Questionnaire", "Voter Registration Form", "Passport Application (DS-11)", "Social Security Card Application",
    "Change of Address Form", "FOIA Request", "Dog License Application", "Gun Permit Application",
    "Marriage License Application", "Death Certificate Worksheet", "IRS Form 1040", "W-4 Form",
    "VAT Return Form", "Customs Declaration Form", "Currency Transaction Report (CTR)", "Loan Application (URLA)",
    "Deposit Slip", "Wire Transfer Request", "Beneficiary Designation Form", "Expense Reimbursement Form",
    "New Patient Intake Form", "HIPAA Release Form", "Surgical Consent Form", "DNR Order",
    "PHQ-9 (Patient Health Questionnaire)", "Vaccination Record Card", "Prescription Pad (Rx)", "Hospital Incident Report",
    "Organ Donor Registration", "Explanation of Benefits (EOB)", "College Application", "FAFSA",
    "Transcript Request Form", "IEP (Individualized Education Program)", "Add/Drop Slip", "Teacher Evaluation Form",
    "Library Card Application", "Thesis Submission Form", "Field Trip Permission Slip", "Alumni Donation Pledge",
    "Job Application", "I-9 (Employment Eligibility Verification)", "Non-Disclosure Agreement (NDA)", "Performance Review",
    "Time Sheet", "Leave Request Form", "Exit Interview Form", "OSHA 301 Accident Report",
    "Direct Deposit Authorization", "Sexual Harassment Complaint Form", "Summons", "Power of Attorney",
    "Last Will and Testament", "Divorce Petition", "Copyright Registration Form", "Eviction Notice",
    "Affidavit", "Plea Form", "Restraining Order Application", "Articles of Incorporation",
    "Residential Lease Agreement", "Property Condition Disclosure", "Building Permit Application", "Home Inspection Report",
    "Mechanic's Lien", "Renters Insurance Application", "HOA Violation Notice", "Zoning Variance Request",
    "Move-In/Move-Out Checklist", "Title Deed", "Visa Application", "Flight Plan",
    "Bill of Lading", "Vehicle Title Transfer", "Traffic Ticket (Citation)", "Lost Baggage Claim",
    "International Driving Permit Application", "Hazmat Shipping Paper", "CDL Medical Report", "Car Rental Agreement",
    "Bug Report Ticket", "Domain Name Registration", "SSL Certificate Request", "Account Recovery Form",
    "DMCA Takedown Notice", "Beta Tester Feedback Form", "API Key Request", "GDPR Data Subject Request",
    "Server Maintenance Log", "Two-Factor Authentication Setup", "Baptismal Certificate", "Scorecard (Golf/Bowling)",
    "Model Release Form", "Raffle Ticket Stub", "Recipe Submission Form", "Tattoo Consent & Waiver",
    "Dive Log", "Chain of Custody Form", "Film Permit Application", "Patent Application"
]

PROMPT = '''Generate exactly {count} realistic forms based on these form types: {form_types}

Output a JSON list of exactly {count} objects. Each object must have these EXACT keys:
- form_id (string, use the IDs provided: {ids})
- form_name (string, the exact form type name from the list)
- description (string, a brief 1-sentence description of what the form is for)
- schema (JSON object where keys are field names and values are questions to collect that field. Include 5-10 relevant fields per form)

Example format:
{{
  "form_id": "67001",
  "form_name": "Census Questionnaire",
  "description": "A mandatory government survey used to count the population and gather demographic data.",
  "schema": {{
    "household_size": "How many people live in this household?",
    "primary_resident_name": "What is the name of the primary resident?",
    "date_of_birth": "What is your date of birth?",
    "ethnicity": "What is your ethnicity?",
    "employment_status": "What is your current employment status?"
  }}
}}

Make each form unique with appropriate fields for its purpose. Be precise and realistic.'''

def validate_form(form):
    required = ["form_id", "form_name", "description", "schema"]
    if not all(k in form for k in required):
        return False
    if not isinstance(form["schema"], dict) or len(form["schema"]) < 3:
        return False
    return True

def generate_batch(model, batch_size, start_id, form_types_subset):
    ids = [str(67001 + start_id + i) for i in range(batch_size)]
    prompt = PROMPT.format(count=batch_size, form_types=", ".join(form_types_subset), ids=", ".join(ids))
    
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            if not isinstance(data, list):
                print(f"  Attempt {attempt + 1}: Response is not a list")
                time.sleep(5)
                continue
            if len(data) != batch_size:
                print(f"  Attempt {attempt + 1}: Got {len(data)} items, expected {batch_size}")
                time.sleep(5)
                continue
            invalid = [i for i, f in enumerate(data) if not validate_form(f)]
            if invalid:
                print(f"  Attempt {attempt + 1}: Invalid forms at indices {invalid}")
                time.sleep(5)
                continue
            return data
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON decode error: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(60)
            else:
                time.sleep(10)
    raise Exception(f"Failed to generate valid batch after retries")

def generate_forms():
    filepath = "data/generated_forms.json"
    existing = json.load(open(filepath)) if os.path.exists(filepath) else []
    start_idx = len(existing)
    
    if start_idx >= 100:
        print(f"Already have {start_idx} forms, done!")
        return
    
    print(f"Continuing from form {start_idx + 1}/100")
    
    api_keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_2")]
    models = ["models/gemini-2.5-flash", "models/gemini-1.5-flash"]
    
    for api_key in api_keys:
        if start_idx >= 100:
            break
        genai.configure(api_key=api_key)
        for model_name in models:
            if start_idx >= 100:
                break
            print(f"Using model: {model_name}")
            model = genai.GenerativeModel(model_name, 
                                          generation_config={"response_mime_type": "application/json", "temperature": 1.0})
            
            while start_idx < 100:
                try:
                    batch_size = min(5, 100 - start_idx)
                    form_types_subset = FORM_TYPES[start_idx:start_idx + batch_size]
                    
                    print(f"Generating forms {start_idx + 1} to {start_idx + batch_size}...")
                    batch = generate_batch(model, batch_size, start_idx, form_types_subset)
                    existing.extend(batch)
                    
                    with open(filepath, 'w') as f:
                        json.dump(existing, f, indent=2)
                    
                    start_idx = len(existing)
                    print(f"✓ Saved {start_idx}/100 forms")
                    time.sleep(2)
                except Exception as e:
                    print(f"Switching to next model/key due to: {str(e)[:100]}")
                    break
    
    print("Complete! Generated 100 forms.")

if __name__ == "__main__":
    generate_forms()
