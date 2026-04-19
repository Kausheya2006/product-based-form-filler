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
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```

## Model switching (2-line quick use)
In `src/.env`, set exactly one path: `USE_MODAL_INFERENCE=true` (cloud Qwen3), or `USE_LOCAL_CONTAINER_GEMMA4=true` (localhost Gemma container), or `USE_OLLAMA=true` (default local Ollama); keep others `false` and `MODEL_SERVICE_URL=` empty; for Modal SDK mode, run `modal setup` once (or use Modal token env vars).  
Backend priority is fixed: `USE_MODAL_INFERENCE` → `USE_LOCAL_CONTAINER_GEMMA4` → `MODEL_SERVICE_URL` → `USE_OLLAMA` → bundled local model.

## Testing

Run all tests from `src/`.

### Interface tests (fast)

```sh
export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
export MOCK_MODELS="true"
python -m pytest tests/test_interface_flows.py
```

Run one test case:

```sh
python -m pytest tests/test_interface_flows.py -k tc08
```

### E2E tests (Playwright)

Install browser once:

```sh
python -m playwright install chromium
```

Start app in one terminal:

```sh
export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```

Run E2E in another terminal:

```sh
bash scripts/run_e2e_headed.sh
# or
bash scripts/run_e2e_headless.sh
```


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
