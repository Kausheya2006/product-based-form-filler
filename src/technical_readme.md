# Technical README

This document explains the runtime architecture, domain models, AI model backends, and design choices in the ProductLabs AI Form Filler codebase.

## 1. System Overview

At a high level, the project is a FastAPI application that:

- stores forms, conversations, outputs, users, and run logs in MongoDB
- accepts conversations as text, live entry, or audio
- runs extraction and summarization through a pluggable model layer
- renders a server-side HTML interface using Jinja templates

There are two main runtime shapes:

1. A single `app` process that loads models directly or calls Ollama/Modal.
2. A split deployment where the web app calls a separate `model-service` FastAPI process over HTTP.

## 2. Top-Level Runtime Components

```text
Browser
  -> FastAPI app (`src.interface.api`)
     -> handlers/helpers (`src.interface.helpers`)
        -> repositories (`src.infrastructure.persistence.mongo`)
        -> pipeline (`src.application.pipeline.FormFillingService`)
           -> extraction model backend
           -> summarizer backend
        -> ASR / translation / diarization helpers
  -> MongoDB

Optional:
app -> model-service (`src.interface.model_service_api`) -> local model classes
```

## 3. Layered Structure

The project follows a clean-architecture-inspired layout.

```text
src/src/
├── domain/         # core data models and abstract interfaces
├── application/    # orchestration and business rules
├── infrastructure/ # persistence, config, model implementations
└── interface/      # FastAPI routes, HTML handlers, WebSocket collaboration
```

### `domain/`

Purpose:
- define the main business objects
- define repository and pipeline contracts
- keep framework-specific code out of core models

Key files:
- `domain.py`
- `interfaces.py`
- `speakers.py`

### `application/`

Purpose:
- orchestrate extraction runs
- validate and normalize extracted values
- create run-log records and status updates

Key file:
- `pipeline.py`

### `infrastructure/`

Purpose:
- connect the app to MongoDB
- load local models
- call remote model backends
- hold configuration from environment variables

Key folders:
- `persistence/`
- `ai/`

### `interface/`

Purpose:
- expose HTTP routes and WebSocket endpoints
- render server-side templates
- handle auth, forms, conversations, outputs, and ASR flows

Key files:
- `api.py`
- `helpers.py`
- `collab_ws.py`
- `model_service_api.py`
- `dependencies.py`

## 4. Composition Root and Dependency Wiring

`src/src/interface/dependencies.py` is the composition root.

It initializes:

- Mongo repositories
- the extraction model backend
- the summarizer backend
- the application pipeline
- translator, ASR, STT, and diarizer helpers

The model backend is chosen at startup from environment flags. The selection precedence is:

```text
MOCK_MODELS
-> USE_MODAL_INFERENCE
-> USE_LOCAL_CONTAINER_GEMMA4
-> MODEL_SERVICE_URL
-> USE_OLLAMA
-> bundled local models
```

This means the app can be switched between mock, remote, local-container, Ollama, Modal, and bundled local inference without changing handler code.

## 5. Request and Processing Flows

## 5.1 Form creation flow

1. User opens `/forms/new`.
2. `FormHandler.create_form` validates input and builds a `FormSchema`.
3. The form is persisted through `MongoFormRepository`.

Important detail:
- On the create page, field values are chosen from strict types like `string`, `int`, `email`, and `date`.
- On the edit page, those same schema values can be changed into natural-language instructions.

So, in practice, `FormSchema.fields` acts as both:
- a type-hint store for validation
- an extraction prompt or instruction store for richer schemas

## 5.2 Static text extraction flow

1. A conversation is entered in `Speaker: text` format.
2. `ConversationParser.parse` converts it into ordered internal history.
3. `ConversationHandler.create_conversation` saves the conversation.
4. `FormFillingService.run` loads the conversation and form.
5. The pipeline calls the configured extraction model and summarizer.
6. Results are validated, stored in `run_logs`, and saved to `outputs`.

## 5.3 Live extraction flow

The live path uses `_extract_for_conversation_text` in `helpers.py`.

Behavior:
- parses the current conversation text
- seeds current form state
- calls `process_live_update(...)` when the backend supports it
- optionally accepts out-of-schema field suggestions
- merges accepted new fields into the displayed result

This path is important because the default extraction design is incremental rather than purely one-shot.

## 5.4 Audio extraction flow

For audio uploads:

1. Audio is transcribed.
2. If needed, text is translated to English.
3. Optional diarization assigns speaker labels.
4. The resulting conversation text is converted into the same internal format as text input.
5. Extraction is replayed line by line using the live-update model contract.

This keeps audio extraction aligned with the same downstream form-state logic used by the live text workflow.

## 5.5 Collaborative conversation flow

`collab_ws.py` provides a room-based WebSocket manager.

Characteristics:
- rooms are keyed by `room_id`
- multiple users receive message, typing, join, and leave events
- collaborative entry is real-time at the UI/WebSocket layer

The WebSocket layer does not itself replace the persistence model. Conversations still become durable only when saved through the normal handlers.

## 6. Domain Models

These are the core data objects in `src/src/domain/domain.py`.

## `ConversationVersion`

Represents one saved version of a conversation.

Fields include:
- `version_index`
- `timestamp`
- `history`
- `run_id`
- `source_mode`
- `input_language`
- `raw_transcript`
- `translated_transcript`

Why it matters:
- conversations are versioned rather than overwritten
- ASR metadata stays attached to the specific saved version

## `Conversation`

Represents a conversation associated with a form.

Key behaviors:
- stores multiple versions
- exposes `latest_history`
- exposes `full_text`, which renders the latest history into model-friendly text

## `FormSchema`

Represents an extractable form template.

Key fields:
- `form_id`
- `form_name`
- `description`
- `schema`
- `visibility`
- `collaborators`
- `owner_id`

Important detail:
- `schema` is a flat mapping of `field_name -> value`
- the value may be a strict type or a natural-language instruction

## `ExtractionResult`

Represents the output of one extraction run.

Key fields:
- `conversation_id`
- `form_id`
- `filled_data`
- `accepted_new_fields`
- `run_id`
- `summary`

## `ExtractionRequest`

Represents a per-field extraction request used by batch-style extractors.

Fields:
- `context`
- `field_name`
- `instruction`
- `original_type_hint`

## `RunLog`

Tracks the lifecycle of a pipeline run.

Fields include:
- `run_id`
- `conversation_id`
- `version_index`
- `started_at`
- `finished_at`
- `status`
- `error`
- `summary`
- `extracted_fields`
- `owner_id`

This is the audit trail behind the admin run log pages.

## 7. Domain Interfaces

`src/src/domain/interfaces.py` defines the boundary contracts:

- `IConversationRepository`
- `IFormRepository`
- `IExtractionModel`
- `IPipeline`
- `IRunLogRepository`
- `ISummarizer`

This keeps the application layer independent of specific storage and model implementations.

## 8. Persistence Design

MongoDB persistence lives in `src/src/infrastructure/persistence/mongo.py`.

Collections used by the app:

- `forms`
- `conversations`
- `run_logs`
- `outputs`
- `users`

Repository responsibilities:

- `MongoConversationRepository`
  - fetch/save conversations by `conversation_id`
- `MongoFormRepository`
  - fetch/save/delete forms by `form_id`
- `MongoRunLogRepo`
  - create, update, list, and fetch run logs
  - ensures indexes for `run_id`, `started_at`, and `(conversation_id, version_index)`

## 9. Conversation Representation

The internal conversation representation is slightly unusual but intentional.

Input text:

```text
Doctor: Hello
Patient: I am Alice
```

Internal `history` shape:

```python
{
  "Doctor 000001": "Hello",
  "Patient 000002": "I am Alice",
}
```

Why this exists:
- preserves insertion order
- supports repeated speakers without key collisions
- allows the rendering helper to strip numeric suffixes before sending text to the model

`domain/speakers.py` is responsible for normalizing and rendering this structure.

## 10. Pipeline Design

The main orchestration class is `FormFillingService` in `application/pipeline.py`.

Responsibilities:

- create a `run_id`
- write a `running` run log
- load the conversation and form
- gather extraction and summarization
- validate extracted values against expected types
- persist success or failure status in `run_logs`

### Type normalization

`validate_field_type(...)` currently supports:

- `string`
- `int`
- `float`
- `email`
- `phone`
- `date`

If a schema value is not one of these known types, the value is accepted as-is. That is what makes it possible for edited forms to use free-form extraction instructions.

### Incremental extraction design

When the pipeline runs in `full_process` mode, it:

- starts with all form fields seeded as `N/A`
- renders the latest conversation version into lines
- replays the conversation one line at a time
- repeatedly calls `process_live_update(...)`
- carries forward the running field state

This is one of the key architectural ideas in the project: extraction is modeled as evolving form state, not only as one final batch pass.

## 11. Extraction and Summarization Backends

The `infrastructure/ai/` folder contains several interchangeable backends.

## 11.1 Extraction backends

### `MockExtractionModel`

Purpose:
- zero-ML fallback for development and testing

Behavior:
- returns `N/A` for requested fields
- returns one dummy suggested field in live mode

### `RemoteModelServiceExtractionModel`

Purpose:
- call a separate FastAPI model-service over HTTP

Behavior:
- POSTs to `/extract` or `/live-extract`
- falls back to the mock model if the remote service is unavailable

### `ModalExtractionModel`

Purpose:
- run extraction on Modal

Modes:
- SDK mode using `modal.Function.from_name(...)`
- HTTP mode using a configured endpoint

Behavior:
- supports both one-shot extraction and live-update extraction
- falls back to mock on failure

### `OllamaFormStateModel`

Purpose:
- use an Ollama-served LLM for extraction

Behavior:
- for generic models, it can do simple extraction from a prompt
- for Qwen-based models, it asks for structured JSON containing:
  - `filled_data`
  - `suggested_new_fields`

This is the backend that most explicitly supports schema extension suggestions during live extraction.

### `FormStateModel`

Purpose:
- bundled local incremental extractor using a LoRA adapter

Implementation notes:
- loads a tokenizer and base causal model
- optionally attaches a PEFT adapter
- builds chat messages with current summary, current form state, conversation context, and the new line
- expects tool-call-shaped JSON that updates:
  - next form state
  - summary state

This is the most important local extraction model in the current design.

### `GemmaFunctionalModel`

Purpose:
- older or alternate local extraction path using a local checkpoint

Behavior:
- performs one-shot causal generation
- extracts a JSON object from the response

### `LocalHuggingFaceModel`

Purpose:
- baseline extractive QA model using the transformers question-answering pipeline

Behavior:
- performs batched field extraction by asking one question per field
- uses a confidence threshold to suppress low-confidence spans

This is the simplest extraction design in the repository, but it is not the main runtime path anymore.

## 11.2 Summarization backends

### `MockSummarizer`

- returns a placeholder summary

### `RemoteModelServiceSummarizer`

- calls the model-service `/summarize` endpoint
- falls back to mock on failure

### `ModalSummarizer`

- supports Modal SDK or HTTP mode

### `OllamaSummarizer`

- summarizes through Ollama with a short-prompt interface

### `LocalSummarizer`

- uses `sshleifer/distilbart-cnn-12-6`
- standard seq2seq summarization

### `GemmaSummarizer`

- incremental summarizer
- processes the conversation line by line
- feeds the current summary back into the next prompt

### `QwenSummarizer`

- one-shot summarizer using `Qwen/Qwen2.5-1.5B-Instruct`
- currently the default local summarizer path in config

## 12. Audio and Language Helpers

The app includes a separate set of speech-related helpers.

### `LocalASRTranscriber`

Purpose:
- Whisper-based transcription for uploaded or recorded audio

Model:
- `openai/whisper-small`

### `LocalSpeechToText`

Purpose:
- alternate STT path used by live audio preview endpoints

Pipeline:
- normalize audio
- simple energy-based VAD
- noise reduction
- Google SpeechRecognition backend
- basic inverse text normalization for number phrases

Important note:
- the commented Whisper path is currently disabled in this class

### `LocalSpeakerDiarizer`

Purpose:
- split audio by speaker

Pipeline:
- Whisper timestamped ASR
- pyannote speaker embeddings
- agglomerative clustering
- merge consecutive same-speaker segments

### `LocalTranslator`

Purpose:
- translate supported non-English transcripts to English before extraction

Supported source languages:
- Spanish
- French
- German
- Italian
- Portuguese

## 13. Interface Layer Design

The web layer is split across `api.py`, `helpers.py`, templates, static JS, and WebSocket code.

`api.py`:
- registers routes
- creates the FastAPI app
- mounts static files
- initializes repositories and indexes during lifespan startup

`helpers.py`:
- contains most service and handler logic
- centralizes auth, permissions, parsing, extraction, output persistence, and handler methods

This file is large, but it acts as the main application-facing façade for the route layer.

Important helper classes:

- `AuthService`
- `UserRepository`
- `AccessPolicy`
- `FormQueryService`
- `ConvoQueryService`
- `ConversationParser`
- `SchemaBuilder`
- `FieldMerger`
- `ExtractionService`
- `OutputRepository`
- `TemplateRenderer`
- `AuthHandler`
- `AdminHandler`
- `FormHandler`
- `ConversationHandler`
- `ExtractionHandler`
- `ASRHandler`
- `OutputHandler`

## 14. Access Control Model

The access rules are centered in `AccessPolicy`.

Rules in practice:

- admins can read and write everything
- personal objects are owned by a single user
- global forms are visible to all users
- collaborative forms allow named collaborators to use the form
- conversation write access remains owner-focused

This means form visibility and conversation ownership are related but not identical concepts.

## 15. Model-Service Subsystem

`src/src/interface/model_service_api.py` exposes a smaller FastAPI app that can host local models behind HTTP.

Endpoints:

- `GET /health`
- `POST /extract`
- `POST /live-extract`
- `POST /summarize`

Why it exists:

- isolates heavy model loading from the main web app
- allows the UI process to stay lighter
- creates a clean HTTP contract for extraction and summarization

## 16. Configuration

Configuration lives in `src/src/infrastructure/config.py` using `pydantic-settings`.

Important settings:

- `MONGO_URI`
- `DB_NAME`
- `MOCK_MODELS`
- `MODEL_SERVICE_URL`
- `USE_OLLAMA`
- `USE_MODAL_INFERENCE`
- `USE_LOCAL_CONTAINER_GEMMA4`
- `EXTRACTION_MODEL_TYPE`
- `FORM_STATE_MODEL_PATH`
- `SUMMARIZER_TYPE`
- `SUMMARIZER_MODEL_PATH`

The compose stack supplies defaults for the app and model-service containers, but the true runtime behavior comes from these settings.

## 17. Testing Structure

Test areas in the repo:

- `tests/test_interface_flows.py`
  - fast interface tests
  - focus on routes and user flows
- `tests/e2e/test_ui_flows.py`
  - Playwright browser tests
  - exercise registration, form creation, extraction pages, profile updates, and navigation

Scripts in `scripts/` wrap common pytest and Playwright commands.

## 18. Design Strengths

- Clean separation between domain, application, infrastructure, and interface concerns.
- Pluggable inference backends with a single pipeline contract.
- Versioned conversations make re-extraction and auditing easier.
- Separate run logs and outputs give both auditability and user-facing history.
- Audio, translation, and diarization are integrated into the same downstream extraction model.

## 19. Design Tradeoffs and Caveats

- `helpers.py` is doing a lot of work and is the largest concentration of orchestration logic.
- Some backends silently fall back to mock behavior, which is useful for resilience but can hide misconfiguration.
- Schema values serve double duty as type hints and instructions, which is flexible but conceptually mixed.
- Collaborative form access and conversation ownership rules are not identical, so permission behavior needs careful reasoning.

## 20. Best Extension Points

If you want to extend the system, these are the best places to start:

### Add a new extraction backend

- implement the `IExtractionModel` contract in `infrastructure/ai/`
- wire it in `interface/dependencies.py`

### Add a new summarizer backend

- implement `ISummarizer`
- wire it in `interface/dependencies.py`

### Change extraction orchestration

- edit `application/pipeline.py`

### Change form/conversation persistence

- edit `infrastructure/persistence/mongo.py`
- or add a new repository implementation behind the existing interfaces

### Change user-facing flows

- edit `interface/helpers.py`
- update templates in `interface/templates/`

## 21. Practical Mental Model

If you need one concise way to think about the system, it is this:

The application stores versioned conversations and form schemas in MongoDB, turns conversations into a running form state through a pluggable extraction backend, summarizes the same conversation in parallel, and exposes the whole flow through a server-rendered FastAPI UI with optional audio and collaboration features.
