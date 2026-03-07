import google.generativeai as genai
import os, json, time, random, uuid
from dotenv import load_dotenv

load_dotenv()

FILEPATH = "data/generated_conversations.json"
FORMS_PATH = "data/generated_forms.json"
CONVOS_PER_FORM = 5
TARGET = 500

# 8 distinct scenario archetypes — {form_name} substituted at runtime
SCENARIO_ARCHETYPES = [
    "Two people: a professional or official and the subject/applicant going through the {form_name} in a formal setting. Use roles specific to this form (e.g. doctor/patient, officer/applicant, agent/client).",
    "A single person alone, thinking aloud or speaking into a voice recorder while working through the {form_name} step by step.",
    "A phone or video call where one party collects {form_name} details from the other — include natural pauses, hold music, re-asking of unclear answers, and filler.",
    "Three or more people — pick roles that fit the {form_name} (e.g. family member helping, professional asking questions, applicant answering) with cross-talk and interruptions.",
    "A formal interview, official questioning, or structured intake where {form_name} details emerge through directed Q&A. The interviewer controls the pace firmly.",
    "A stressful or emotionally charged scene — an accident, dispute, urgent deadline, or difficult life event — where the {form_name} must be completed under pressure.",
    "A casual scene: a friend, sibling, or parent helping someone fill in the {form_name}, with digressions, jokes, and tangents before getting back on track.",
    "An in-person counter or desk visit — a receptionist or clerk processes the {form_name} with the person across the counter, with queue noise and paperwork shuffling.",
]

# Extras injected by Python — not decided by the LLM
EXTRA_RELEVANT = "Naturally include a few extra details about the situation or person that are NOT covered by any form field (realistic side info)."
EXTRA_IRRELEVANT = "Weave in one brief unrelated tangent or small talk moment that briefly breaks the flow before returning to the topic."

# Single-convo prompt — used when generating leftover (< 5) convos for a partially-done form
PROMPT_ONE = '''Generate a realistic conversation for the following form. This is synthetic training data for an AI that extracts form fields from conversations.

Form: {form_name}
Purpose: {form_description}
Fields (cover most, not all): {schema_fields}

Scenario: {scenario}
Length: {min_words}–{max_words} words.{extras}

Rules:
- Invent realistic names, dates, addresses — no placeholders.
- Speaker labels are role-appropriate and consistent (e.g. "Dr. Patel", "Officer Brooks", "Lena").
- Append a unix timestamp to each speaker label (start ~1739900000, increment 5–30s per turn).
- Make it feel human: hesitations, re-asks, corrections, small courtesies.
- Form fields emerge organically — not a rigid checklist.

Output ONLY a flat JSON object: keys = "SpeakerLabel UnixTimestamp", values = spoken text. No markdown, no wrapper.

Example shape:
{{"Dr. Patel 1739900000": "Good morning, take a seat.", "Marcus 1739900018": "Thanks.", "Dr. Patel 1739900031": "Let's start with your full name."}}'''

# Batch prompt — generates all 5 conversations for one form in a single API call
PROMPT_BATCH = '''Generate exactly {count} distinct realistic conversations for the form below. This is synthetic training data for an AI that extracts form fields from conversations.

Form: {form_name}
Purpose: {form_description}
Fields (cover most, not necessarily all, in each convo): {schema_fields}

Each conversation must use a DIFFERENT scenario from this list (use them in order):
{scenarios}

Per-conversation lengths (use in order): {lengths}
Per-conversation extras (apply only where marked, otherwise none): {extras_list}

Rules for ALL conversations:
- Invent realistic names, dates, addresses — no placeholders ever.
- Speaker labels are role-appropriate and consistent per conversation (e.g. "Dr. Patel", "Officer Brooks", "Lena").
- Append a unix timestamp suffix to each speaker label. Each conversation starts at a different unix time (~1739900000 + random offset per convo). Increment 5–30s per turn.
- Make it human: hesitations, re-asks, corrections, courtesies. Form fields emerge organically.

Output ONLY a JSON object with exactly {count} keys: "1", "2", ..., "{count}". Each value is a flat dict of "SpeakerLabel UnixTimestamp" → spoken text for that conversation. No markdown, no extra keys.

Example (2 conversations):
{{
  "1": {{"Clerk 1739900000": "Next please!", "Applicant 1739900010": "Hi, I need to file this.", "Clerk 1739900022": "Sure, what is your full name?"}},
  "2": {{"Officer 1739910000": "Good afternoon, please state your name.", "Suspect 1739910015": "Marcus Webb.", "Officer 1739910030": "And your date of birth?"}}
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
        # Model may have returned N separate JSON objects instead of an array.
        # Try splitting on top-level object boundaries.
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
    pool = SCENARIO_ARCHETYPES[:]
    random.shuffle(pool)
    return [s.format(form_name="{form_name}") for s in pool[:n]]


def make_extras_list(n):
    result = []
    for _ in range(n):
        parts = []
        if random.random() < 0.10:
            parts.append(EXTRA_RELEVANT)
        if random.random() < 0.10:
            parts.append(EXTRA_IRRELEVANT)
        result.append(" ".join(parts) if parts else "none")
    return result


LENGTH_CHOICES = [(500, 1200), (800, 2000), (1500, 3000), (2500, 5000)]


def call_batch(model, form, count):
    """One API call → list of count conversation dicts."""
    scenarios = [s.format(form_name=form["form_name"]) for s in pick_n_scenarios(count)]
    lengths = [random.choice(LENGTH_CHOICES) for _ in range(count)]
    extras_list = make_extras_list(count)
    schema_fields = ", ".join(form.get("schema", {}).keys())

    prompt = PROMPT_BATCH.format(
        count=count,
        form_name=form["form_name"],
        form_description=form.get("description", ""),
        schema_fields=schema_fields,
        scenarios="\n".join(f"{i+1}. {s}" for i, s in enumerate(scenarios)),
        lengths=", ".join(f"{mn}–{mx} words" for mn, mx in lengths),
        extras_list=", ".join(f"[{i+1}] {e}" for i, e in enumerate(extras_list)),
    )

    for attempt in range(4):
        try:
            response = model.generate_content(prompt)
            data = parse_json_response(response.text)
            # Expect dict with keys "1".."count", each value a conversation dict
            if isinstance(data, dict):
                convos = [data[str(i+1)] for i in range(count) if str(i+1) in data]
                # Accept if each value is a dict — allow single-turn (model may be conservative)
                valid = [c for c in convos if isinstance(c, dict) and len(c) >= 1]
                if len(valid) == count:
                    return valid
            print(f"    Bad shape attempt {attempt+1} (type={type(data).__name__}, "
                  f"keys={list(data.keys())[:5] if isinstance(data, dict) else len(data) if isinstance(data, list) else '?'}), retrying...")
            time.sleep(6)
        except json.JSONDecodeError as e:
            print(f"    JSON error attempt {attempt+1}: {e}")
            time.sleep(8)
        except Exception:
            raise
    return []


def call_single(model, form, scenario):
    """Fallback single-convo call for partial resumes."""
    min_w, max_w = random.choice(LENGTH_CHOICES)
    extras_parts = []
    if random.random() < 0.10:
        extras_parts.append(EXTRA_RELEVANT)
    if random.random() < 0.10:
        extras_parts.append(EXTRA_IRRELEVANT)
    extras = ("\n- " + "\n- ".join(extras_parts)) if extras_parts else ""

    prompt = PROMPT_ONE.format(
        form_name=form["form_name"],
        form_description=form.get("description", ""),
        schema_fields=", ".join(form.get("schema", {}).keys()),
        scenario=scenario.format(form_name=form["form_name"]),
        min_words=min_w,
        max_words=max_w,
        extras=extras,
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
        pass  # ensure file exists
    # write full array (append-safe: rewrite whole file since it's JSON array)
    with open(FILEPATH, "w") as f:
        json.dump(existing, f, indent=2)


def generate_conversations():
    forms = json.load(open(FORMS_PATH))
    existing = json.load(open(FILEPATH)) if os.path.exists(FILEPATH) else []
    counts = get_counts(existing)
    total = sum(counts.values())

    if total >= TARGET:
        print(f"Already at {total}/{TARGET}. Done.")
        return

    print(f"Starting from {total}/{TARGET}")

    for api_key, model_name in build_pairs():
        pending = [f for f in forms if counts.get(f["form_id"], 0) < CONVOS_PER_FORM]
        if not pending:
            print("All 500 done!"); break

        print(f"\n── {model_name}  key …{api_key[-6:]}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name, generation_config={"temperature": 1.1})
        daily_exhausted = False

        for form in pending:
            if daily_exhausted:
                break
            fid = form["form_id"]
            needed = CONVOS_PER_FORM - counts.get(fid, 0)

            try:
                if needed == CONVOS_PER_FORM:
                    # Full batch: 1 API call for all 5
                    convos = call_batch(model, form, CONVOS_PER_FORM)
                    if convos:
                        docs = [wrap_doc(c, fid) for c in convos]
                        save(existing, docs, fid, counts)
                        total = sum(counts.values())
                        print(f"  ✓ {total}/{TARGET}  form {fid}  (5/5) [batch]")
                    else:
                        print(f"  ✗ batch failed for {fid}")
                else:
                    # Partial: individual calls for the remaining
                    scenarios = pick_n_scenarios(needed)
                    for scenario in scenarios:
                        convo = call_single(model, form, scenario)
                        if convo:
                            save(existing, [wrap_doc(convo, fid)], fid, counts)
                            total = sum(counts.values())
                            print(f"  ✓ {total}/{TARGET}  form {fid}  ({counts[fid]}/{CONVOS_PER_FORM}) [single]")
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
    print(f"\nAll 9 pairs exhausted. Conversations saved: {total}/{TARGET}")


if __name__ == "__main__":
    generate_conversations()
