# data_generation — README

This folder contains helper scripts that synthesize conversations, form schemas, and labeled datapoints used to train and evaluate the extraction pipeline. Most scripts call the Google/`genai` API (via `google.generativeai`) and read/write JSON files under the repository `data/` directory.

Overview / common conventions
- All generator scripts expect a `.env` with one or more `GEMINI_API_KEY*` variables (used by `genai.configure`).
- Input/Output live in `../data/` (e.g. `generated_forms.json`, `generated_conversations.json`, `generated_datapoints.json`, etc.).
- Prompts are carefully templated strings; responses are parsed with robust JSON-extraction helpers because model outputs sometimes include code fences or stray text.
- Files use simple append-or-rewrite patterns: generators attempt to be idempotent by checking existing outputs and skipping processed items.

File summaries (technical details)

- `generate_form_fields.py`: Generate form schemas (master form catalogue)
  - Purpose: create a curated list of form templates (IDs, names, descriptions, and a `schema` mapping of field_name → question).
  - How it works: uses `genai.GenerativeModel` to request JSON arrays of form objects (batched, up to 5 per call). Each object must include `form_id`, `form_name`, `description`, and a `schema` dict.
  - Outputs: writes `data/generated_forms.json` (list of form objects). Performs validation on shape and retries/rotates models on errors or rate limits.

- `generate_conversations.py`: Produce long, realistic conversations per form (batch mode for many tokens)
  - Purpose: synthesize long-form conversations (training data) that contain the information required by a form.
  - How it works: templates two prompt modes: `PROMPT_BATCH` (single API call generates 5 convos for a form) and `PROMPT_ONE` (fallback single convo). It shuffles scenario archetypes and controls length ranges.
  - I/O: reads `data/generated_forms.json`, appends conversation documents to `data/generated_conversations.json` as objects with `conversation_id`, `form_id`, `conversation` (flat dict `"Speaker TIMESTAMP": text`) and `versions` history.
  - Robustness: `parse_json_response` extracts JSON from noisy responses; quotas trigger model/key rotation.

- `generate_small_conversations.py`: Short-form conversations (short lines)
  - Purpose: same as `generate_conversations.py` but produces short conversations where each line is <= 10 words — useful for lightweight datapoints, edge-case pipelines, or faster generation.
  - Differences: shorter LENGTH_CHOICES, different file path (`data/generated_small_conversations.json`), same batch/single logic and save format as the main conversations script.

- `generate_small_conversations_edgecase.py`: Short conversations focused on edge cases
  - Purpose: synthesize conversations that explicitly exercise corrections, deletions, retractions, contradictions, and other extraction edge cases.
  - Differences: uses edge-case-specific prompt templates (`PROMPT_ONE`, `PROMPT_BATCH` tuned for corrections), and writes to `data/generated_small_conversations_edgecase.json`.

- `generate_datapoints.py`: Turn conversations into structured training datapoints
  - Purpose: for each (long) conversation, call a model to select N target lines and produce structured training examples describing the `current_form_state`, `next_form_state`, `10_lines_before`, and an explanatory `thinking` field.
  - How it works: builds a heavy prompt (`BASE_PROMPT`) that instructs the model to return exactly `DATAPOINTS_PER_CONVO` JSON objects in a strict array. The script validates shape, injects `_conversation_id` for traceability, and appends results to `data/generated_datapoints.json`.
  - Notes: probabilistic focus selection determines if the call emphasizes insertions, corrections, or filler lines; uses retries and key/model rotation for quota handling.

- `generate_small_datapoints.py`: Datapoints for short conversations
  - Purpose: same pipeline as `generate_datapoints.py` but operates on `data/generated_small_conversations.json` and writes `data/generated_small_datapoints.json`.
  - Differences: prompt and length expectations tuned for short-line conversations; same shape enforcement and append logic.

- `generate_small_datapoints_edgecase.py`: Datapoints specifically for edge-case conversations
  - Purpose: produce labeled datapoints that focus exclusively on corrections/revisions/deletions found in `generated_small_conversations_edgecase.json`.
  - How it works: uses an edge-case `BASE_PROMPT` that instructs the model to pick lines representing edge-case behaviors, returns exact-length arrays, and appends results to `data/generated_small_datapoints.json`.

- `generate_linewise_data.py`: Line-level deltas for each conversation
  - Purpose: produce a line-by-line mapping of which fields change on each line of a conversation (an array aligned with conversation lines). Useful for training sequence labeling or incremental extraction logic.
  - How it works: sorts conversation keys by timestamp, formats a readable `Line X: Speaker: Text` block, and asks the model to return a JSON array with one object per line representing the fields that changed. The script post-processes array lengths (pads/truncates) and saves entries to `data/generated_linewise_data.json`.

- `eval_checkpoint.py`: Local checkpoint inference / quick evaluator
  - Purpose: run a local causal-LM checkpoint against a single example to test model behavior (not a generator that calls external `genai`).
  - How it works: loads a Hugging Face / custom checkpoint via `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`), builds a small prompt from the example's `conversation`, tokenizes and runs `model.generate`, then attempts to parse a JSON object from the generated text.
  - Config: supports `--input-json`, `--checkpoint`, `--max-input-tokens`, `--max-new-tokens`, `--temperature`, and outputs a JSON report containing `generated_text` and decoded `prediction`.
  - Use case: fast local verification of fine-tuned checkpoints or behaviour-driven testing without calling the external API.

How to run (examples)
- Generate forms (creates `data/generated_forms.json`):
  - python data_generation/generate_form_fields.py
- Generate conversations (batch / long):
  - python data_generation/generate_conversations.py
- Generate short / edge-case conversations:
  - python data_generation/generate_small_conversations.py
  - python data_generation/generate_small_conversations_edgecase.py
- Generate datapoints from conversations:
  - python data_generation/generate_datapoints.py
  - python data_generation/generate_small_datapoints.py
  - python data_generation/generate_small_datapoints_edgecase.py
- Generate linewise delta annotations:
  - python data_generation/generate_linewise_data.py
- Evaluate a local model checkpoint on one example:
  - python data_generation/eval_checkpoint.py --input-json data/single_test.json --checkpoint src/data_generation/models/checkpoint-200

Notes and recommendations
- Environment: set GEMINI_API_KEY in `.env` for all generator scripts. The code expects multiple keys optionally for rotation (GEMINI_API_KEY_2, ...).
- Rate limits: scripts implement naive backoff, key rotation, and retries but still can hit quotas — monitor usage and reduce temperature or rate if needed.
- Output validation: each script tries to validate JSON shapes; outputs may still need manual spot-checking. `parse_json_response` helpers are intentionally lax to recover model noise but may silently drop malformed items.
- Extending: prompts are string constants at top of files; changing generation behavior usually involves tuning those templates, token limits, and `generation_config` parameters.
