# ProductLabs AI Form Filler

ProductLabs AI Form Filler is a FastAPI web app that turns conversations into structured form data. Users can create forms, enter or upload conversations, run extraction, review outputs, and manage saved runs.

For architecture and implementation details, see [technical_readme.md](technical_readme.md).

## What You Can Do

- Create forms with fields such as `patient_name`, `email`, `phone`, and `date`.
- Run extraction from typed conversations, live conversation entry, or uploaded audio.
- Review saved outputs and summaries.
- Edit forms and conversations, then re-run extraction.
- Use collaborative forms for shared conversation entry.

## Quick Start With Docker

This is the easiest way to run the full stack.

### Prerequisites

- Docker
- Docker Compose

### Steps

1. Open a terminal in `src/`.
2. Make sure `src/.env` has at least these values:

```env
MONGO_URI=mongodb://mongodb:27017/chat_db
DB_NAME=chat_db
```

3. Start the stack:

```bash
docker compose up --build
```

4. Open the app:

```text
http://localhost:8000
```

### What Starts

- `mongodb` on port `27017`
- `model-service` on port `8001`
- `app` on port `8000`

### Notes

- The first real-model startup can take several minutes because model weights may be downloaded.
- Docker volumes `mongo_data` and `hf_cache` keep database and Hugging Face cache data between runs.
- Stop everything with:

```bash
docker compose down
```

## Local Run Without Full Docker

This is useful if you want to run only MongoDB in Docker and run the FastAPI app directly on your machine.

### Minimal app-only setup

From `src/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.app.txt
docker compose up -d mongodb

export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
export MOCK_MODELS="true"

uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

### Why `MOCK_MODELS=true`?

It lets the app start without loading extraction or summarization models. This is the fastest setup for UI work, flow testing, and general local development.

## Model Runtime Options

The app supports several inference backends. In normal use, choose one mode at a time through `src/.env` or exported environment variables.

### Backend selection order

The app resolves model backends in this order:

```text
MOCK_MODELS
-> USE_MODAL_INFERENCE
-> USE_LOCAL_CONTAINER_GEMMA4
-> MODEL_SERVICE_URL
-> USE_OLLAMA
-> bundled local models
```

### Common modes

#### 1. Fastest local development

```env
MOCK_MODELS=true
```

#### 2. Docker Compose default

Use the bundled `model-service` container:

```env
MODEL_SERVICE_URL=http://model-service:8001
```

This is already the compose default for the `app` container.

#### 3. Ollama

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EXTRACT_MODEL=qwen2.5:1.5b
OLLAMA_SUMMARIZER_MODEL=qwen2.5:1.5b
```

#### 4. Modal

```env
USE_MODAL_INFERENCE=true
MODAL_INFERENCE_USE_SDK=true
MODAL_APP_NAME=monomodel-qwen3-4b-infer
MODAL_EXTRACT_FUNCTION=modal_live_extract
MODAL_SUMMARIZER_FUNCTION=modal_summarize
```

#### 5. Local container Gemma

```env
USE_LOCAL_CONTAINER_GEMMA4=true
LOCAL_CONTAINER_BASE_URL=http://localhost:11434
LOCAL_CONTAINER_EXTRACT_MODEL=gemma4-e2b:latest
LOCAL_CONTAINER_SUMMARIZER_MODEL=gemma4-e2b:latest
```

## Typical User Workflow

1. Register a user account.
2. Create a form.
3. Add fields you want extracted.
4. Choose one of the extraction entry modes:
   - Static text extraction
   - Live extraction
   - Static audio extraction
5. Review the extracted output and summary.
6. Open saved outputs later from the Outputs page.

## Extraction Modes

### Static Text Extraction

- Enter a speaker-labelled conversation such as `Doctor: ...`
- Save the conversation
- Run extraction once
- Review the output page

### Live Extraction

- Enter conversation turns as they happen
- The UI can preview incremental extraction state
- Useful when the conversation grows over time

### Static Audio Extraction

- Upload a conversation recording
- The app transcribes audio
- For supported non-English inputs, it translates to English before extraction
- Optional diarization can split the transcript into speakers

## Running Tests

Run commands from `src/`.

If you are running tests locally outside Docker, install the full test dependencies first:

```bash
pip install -r requirements.txt
docker compose up -d mongodb
```

### Interface tests

```bash
export MOCK_MODELS="true"
python -m pytest tests/test_interface_flows.py # integration tests
python -m pytest tests/test_unit_domain_speakers. # unit tests
python -m pytest tests/test_unit_interface_helpers.py # unit tests
```

Run a single case:

```bash
python -m pytest tests/test_interface_flows.py -k tc08
```

### Browser E2E tests

Install Playwright once:

```bash
python -m playwright install chromium
```

Start the app, then in another terminal:

```bash
bash scripts/run_r2_e2e_headed.sh
```

or

```bash
bash scripts/run_r2_e2e_headless.sh
```

## Troubleshooting

### The app opens but extraction is slow or fails on first run

- Real models may still be downloading.
- Check container logs with `docker compose logs app` and `docker compose logs model-service`.

### Mongo connection errors

- In Docker Compose, use `mongodb://mongodb:27017/chat_db`.
- For local `uvicorn`, use `mongodb://localhost:27017/chat_db`.

### You only want to demo the UI

- Set `MOCK_MODELS=true`.

### Audio features fail

- The app container expects `ffmpeg`.
- Audio and diarization also depend on heavier ML libraries and may take longer to warm up.

## Project Layout

```text
src/
├── docker-compose.yml
├── Dockerfile.app
├── Dockerfile
├── requirements.app.txt
├── requirements.txt
├── scripts/
├── tests/
└── src/
    ├── application/
    ├── domain/
    ├── infrastructure/
    └── interface/
```

## Further Reading

- [technical_readme.md](docs/technical_readme.md) for models, architecture, and design notes
