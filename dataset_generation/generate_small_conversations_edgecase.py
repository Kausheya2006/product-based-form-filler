import google.generativeai as genai
import os, json, time, random, uuid
from dotenv import load_dotenv

load_dotenv()

FILEPATH = "../data/generated_small_conversations_edgecase.json"
FORMS_PATH = "../data/generated_forms.json"
CONVOS_COUNT = 10
FORMS_SAMPLE = 10

USED_FORMS = {"67006", "67012", "67017", "67024", "67028", "67037", "67039", "67043", "67060", "67089"}

EDGE_CASE_SCENARIOS = [
    "A user corrects themselves multiple times during the {form_name} — initially provides wrong info, then catches the mistake and fixes it. Include hesitation and self-correction.",
    "A user deletes or retracts previously given information for the {form_name} — 'Wait, scratch that. Actually that's wrong.' or 'Never mind, let me start over on this field.'",
    "A user changes their mind about information already provided for {form_name} — 'I said X, but now I want to change it to Y' or 'Actually, let me revise that.'",
    "An interviewer questions or challenges information already provided in the {form_name} — forcing the user to re-explain, modify, or acknowledge an error.",
    "A user provides conflicting information for the {form_name} — says one thing early, contradicts it later, then must resolve the conflict.",
    "A user partially gives info for {form_name}, then realizes it's incomplete or incorrect, backfills or revises mid-conversation.",
    "A user undoes or clarifies misleading statements in the {form_name} — 'Actually, that's not quite right. Let me clarify.' or 'That was confusing. Here's what I meant.'",
    "A user requests to remove or clear a field from the {form_name} — 'Can we delete that?' or 'Forget I said that, remove it.'",
]

PROMPT_ONE = '''Generate a realistic SHORT conversation for the following form with PURE EDGE CASES. Keep each line to 10 words maximum. 
This is 100% focused on corrections, deletions, revisions, conflicts, and form field changes — NOT simple single insertions.

Form: {form_name}
Purpose: {form_description}
Fields (cover most, not all): {schema_fields}

Scenario: {scenario}
Length: {min_words}–{max_words} words total.

Rules:
- KEEP EACH LINE TO 10 WORDS OR LESS.
- FOCUS EXCLUSIVELY on edge cases: corrections, deletions, changes, conflicts, retractions, clarifications.
- Include at least 2-3 major revisions or corrections per conversation.
- Invent realistic names, dates, addresses — no placeholders.
- Speaker labels are role-appropriate and consistent.
- Append a unix timestamp to each speaker label.
- Make it feel realistic with hesitations, corrections, and back-and-forth clarification.

Output ONLY a flat JSON object: keys = "SpeakerLabel UnixTimestamp", values = spoken text. No markdown, no wrapper.

Example shape:
{{"Officer 1739900000": "What's your name?", "Suspect 1739900015": "John Smith.", "Officer 1739900030": "Wait, you said John earlier.", "Suspect 1739900045": "Right, John Smith. That's correct."}}'''

PROMPT_BATCH = '''Generate exactly {count} distinct SHORT realistic conversations for the form below with PURE EDGE CASES. Keep each line to 10 words maximum.
This is 100% focused on corrections, deletions, revisions, conflicts, and form field changes — NOT simple single insertions.

Form: {form_name}
Purpose: {form_description}
Fields (cover most, not necessarily all, in each convo): {schema_fields}

Each conversation must use a DIFFERENT scenario from this list (use them in order):
{scenarios}

Per-conversation lengths (use in order): {lengths}

Rules for ALL conversations:
- KEEP EACH LINE TO 10 WORDS OR LESS.
- FOCUS EXCLUSIVELY on edge cases for EACH conversation: corrections, deletions, changes, conflicts, retractions.
- Each conversation must include at least 2-3 major revisions, corrections, or deletions.
- Invent realistic names, dates, addresses — no placeholders ever.
- Speaker labels are role-appropriate and consistent per conversation.
- Append a unix timestamp suffix to each speaker label. Each conversation starts at different unix time. Increment 5–30s per turn.
- Make it realistic with hesitations, corrections, and clarification back-and-forth.

Output ONLY a JSON object with exactly {count} keys: "1", "2", ..., "{count}". Each value is a flat dict of "SpeakerLabel UnixTimestamp" → spoken text. No markdown.

Example (2 conversations):
{{
  "1": {{"Clerk 1739900000": "Name?", "Applicant 1739900010": "John.", "Clerk 1739900022": "Last name?", "Applicant 1739900035": "Smith. Wait, actually Jones.", "Clerk 1739900050": "Jones?", "Applicant 1739900065": "Yes, Jones."}},
  "2": {{"Officer 1739910000": "Date of birth?", "Suspect 1739910015": "May 5, 1990.", "Officer 1739910030": "You said May earlier.", "Suspect 1739910045": "Right. Actually June 3."}}
}}'''


def build_pairs():
    keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_2"), os.getenv("GEMINI_API_KEY_3")]
    models = [
        "models/gemini-2.0-flash", 
        "models/gemini-2.5-flash", 
        "models/gemini-2.5-flash-lite",
        "models/gemini-1.5-flash"
    ]
    return [(k, m) for k in keys if k for m in models]


def get_counts(existing):
    counts = {}
    for c in existing:
        fid = c.get("form_id")
        counts[fid] = counts.get(fid, 0) + 1
    return counts


def is_quota_error(err_str):
    return "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str


def wrap_doc(convo_dict, form_id):
    return {
        "conversation_id": uuid.uuid4().hex[:8],
        "form_id": form_id,
        "conversation": convo_dict,
        "versions": [{"version_index": 0, "timestamp": {"$date": "2026-02-19T00:00:00.000Z"}, "history": convo_dict}]
    }


def parse_json_response(text):
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
        objects = []
        depth, start, in_str, escape = 0, None, False, False
        for i, ch in enumerate(text):
            if escape:
                escape = False; continue
            if ch == '\\' and in_str:
                escape = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objects.append(json.loads(text[start:i+1]))
                    except Exception:
                        pass
                    start = None
        if objects:
            return objects
        raise


def pick_n_scenarios(n):
    pool = EDGE_CASE_SCENARIOS[:]
    random.shuffle(pool)
    return [s.format(form_name="{form_name}") for s in pool[:n]]


LENGTH_CHOICES = [(80, 200), (120, 280), (150, 320)]


def call_single(model, form, scenario):
    """Generate a single edge case conversation."""
    min_w, max_w = random.choice(LENGTH_CHOICES)

    prompt = PROMPT_ONE.format(
        form_name=form["form_name"],
        form_description=form.get("description", ""),
        schema_fields=", ".join(form.get("schema", {}).keys()),
        scenario=scenario.format(form_name=form["form_name"]),
        min_words=min_w,
        max_words=max_w,
    )

    for attempt in range(4):
        try:
            data = parse_json_response(model.generate_content(prompt).text)
            if isinstance(data, dict) and len(data) >= 2:
                return data
            print(f"    Bad single shape attempt {attempt+1}, retrying...")
            time.sleep(6)
        except json.JSONDecodeError as e:
            print(f"    JSON error attempt {attempt+1}: {e}")
            time.sleep(8)
        except Exception:
            raise
    return None


def save(existing, docs, fid, counts):
    for doc in docs:
        existing.append(doc)
        counts[fid] = counts.get(fid, 0) + 1
    with open(FILEPATH, "a") as f:
        pass
    with open(FILEPATH, "w") as f:
        json.dump(existing, f, indent=2)


def generate_edge_case_conversations():
    forms = json.load(open(FORMS_PATH))
    existing = json.load(open(FILEPATH)) if os.path.exists(FILEPATH) else []
    counts = get_counts(existing)
    total = sum(counts.values())

    if total >= CONVOS_COUNT:
        print(f"Already at {total}/{CONVOS_COUNT}. Done.")
        return

    print(f"Starting from {total}/{CONVOS_COUNT}")
    
    available_forms = [f for f in forms if f["form_id"] not in USED_FORMS]
    selected_forms = random.sample(available_forms, min(FORMS_SAMPLE, len(available_forms)))
    print(f"Selected {len(selected_forms)} random forms (avoiding previously used forms) for edge case generation")

    for api_key, model_name in build_pairs():
        pending = [f for f in selected_forms if counts.get(f["form_id"], 0) < 1]
        if not pending:
            print(f"All {CONVOS_COUNT} edge case conversations done!"); break

        print(f"\n── {model_name}  key …{api_key[-6:]}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name, generation_config={"temperature": 1.1})
        daily_exhausted = False

        for form in pending:
            if daily_exhausted:
                break
            fid = form["form_id"]
            needed = 1 - counts.get(fid, 0)

            try:
                if needed == 1:
                    scenarios = pick_n_scenarios(1)
                    for scenario in scenarios:
                        convo = call_single(model, form, scenario)
                        if convo:
                            save(existing, [wrap_doc(convo, fid)], fid, counts)
                            total = sum(counts.values())
                            print(f"  ✓ {total}/{CONVOS_COUNT}  form {fid}  (1/1) [EDGE CASE]")
                        else:
                            print(f"  ✗ single failed for {fid}")
                        time.sleep(2)

                time.sleep(3)

            except Exception as e:
                err = str(e)
                if is_quota_error(err):
                    print(f"  Quota/rate limit on {model_name} → switching pair...")
                    daily_exhausted = True
                else:
                    print(f"  Error: {err[:150]}, skipping form")

    total = sum(counts.values())
    print(f"\nEdge case conversations saved: {total}/{CONVOS_COUNT}")


if __name__ == "__main__":
    generate_edge_case_conversations()
