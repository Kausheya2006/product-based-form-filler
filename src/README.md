# ProductLabs AI Form Filler

Extracts structured form data from conversation logs using NLP.

## Quick Start

**Prerequisites:** Docker & Docker Compose

| Command | Action |
|---------|--------|
| `docker-compose up --build` | First run |
| `docker-compose up` | Start |
| `docker-compose down` | Stop |

Open http://localhost:8000 after `startup complete`

### Steps to run locally : 

```sh
cd src/
docker compose up -d mongodb

export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
export MOCK_MODELS="false"

# in a venv
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```

## Model switching (2-line quick use)
In `src/.env`, set exactly one path: `USE_MODAL_INFERENCE=true` (cloud Qwen3), or `USE_LOCAL_CONTAINER_GEMMA4=true` (localhost Gemma container), or `USE_OLLAMA=true` (default local Ollama); keep others `false` and `MODEL_SERVICE_URL=` empty; for Modal SDK mode, run `modal setup` once (or use Modal token env vars).  
Backend priority is fixed: `USE_MODAL_INFERENCE` → `USE_LOCAL_CONTAINER_GEMMA4` → `MODEL_SERVICE_URL` → `USE_OLLAMA` → bundled local model.

## Testing

This project has two test layers:

- Fast interface tests (CLI, no server/browser/model run)
- Browser E2E tests (Playwright, real UI interactions)

### A) Fast CLI Interface Tests

Run from `src/`:

```sh
python -m pytest -q
```

Run groups:

```sh
python -m pytest -m auth -q
python -m pytest -m profile -q
python -m pytest -m forms -q
python -m pytest -m interface -q
```

### B) Browser E2E Tests (Playwright)

E2E tests live in `tests/e2e/` and are opt-in.

1. Install dependencies once:

```sh
cd src
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

2. Start app in one terminal (example Ollama setup):

```sh
cd src
source .venv/bin/activate
export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
export MOCK_MODELS="false"
export USE_OLLAMA="true"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EXTRACT_MODEL="qwen2.5:1.5b"
export OLLAMA_SUMMARIZER_MODEL="qwen2.5:1.5b"
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```

3. Run E2E from another terminal:

```sh
cd src
source .venv/bin/activate
bash scripts/run_e2e_headed.sh
# or
bash scripts/run_e2e_headless.sh
```

Direct command:

```sh
RUN_E2E=1 APP_BASE_URL="http://127.0.0.1:8000" \
python -m pytest tests/e2e -m e2e --headed --browser chromium \
  --html=reports/e2e_report.html --self-contained-html \
  --tracing=retain-on-failure --video=retain-on-failure --screenshot=only-on-failure
```

Notes:

- `sss` in pytest output means E2E tests were skipped in normal CLI runs.
- E2E HTML reports are written under `src/reports/`.

## Architecture

Clean Architecture (Ports & Adapters) — dependencies point inward only.

```
src/
├── domain/         # Entities + Interfaces (pure Python, no frameworks)
├── application/    # Use cases, orchestration (depends only on domain)
├── infrastructure/ # Implementations: DB, AI models, configs
└── interface/      # Entry points: API routes, DI wiring
```

### Layers

| Folder | What's inside | When to touch |
|--------|---------------|---------------|
| `domain/` | Data models, contracts | Changing what data looks like |
| `application/` | Business logic | Changing how extraction works |
| `infrastructure/` | DB, AI implementations | Swapping tech (new DB, new AI) |
| `interface/` | API routes, startup | Adding endpoints |

### Extending

**Add/Change AI Model:**
```
infrastructure/ai/new_model.py  →  interface/dependencies.py
```

**Add/Change Database:**
```
infrastructure/persistence/new_db.py  →  interface/dependencies.py
```

**Change Extraction Logic:**
```
application/pipeline.py
```

**Add New Data Fields:**
```
domain/domain.py  →  domain/interfaces.py  →  infrastructure/persistence/mongo.py
```
