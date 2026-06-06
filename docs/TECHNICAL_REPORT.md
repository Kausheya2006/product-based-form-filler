# ProductLabs AI Form Filler Architecture and Implementation

This report details the architectural patterns, software domain models, machine learning inference backends, and engineering design tradeoffs implemented in the ProductLabs AI Form Filler platform.

---

## 1. System Overview

The ProductLabs AI Form Filler is a web application framework built on FastAPI designed to process unstructured, multi-turn asynchronous dialogues (text and audio streams) into validated, structured form data. The application maintains domain states, dynamic schemas, execution records, user authentication identities, and run history inside a MongoDB cluster. It executes text extraction and semantic abstractive summarization across a pluggable infrastructure model layer, presenting its states to operators through server-side HTML rendered via Jinja templates.

The application supports two independent operational runtimes:

1. **Monolithic Configuration:** A single Python process managing the FastAPI interface while executing inference tasks locally, over serverless SDK frameworks (Modal Labs), or via container loopbacks (Ollama).
2. **Decoupled Configuration:** A distributed network arrangement where the client-facing application tier sends network requests to a distinct, microservice-isolated `model-service` worker over an explicit HTTP contract.

---

## 2. Top-Level Runtime Components & Architectural Topologies

The codebase decouples its components to separate request processing, state modification execution, and machine learning inference operations.

### 2.1 Architectural Blueprints

#### Core Clean Architecture Boundaries

The layout isolates client-facing components from persistent storage layers and machine learning runends, ensuring code adjustments in external dependencies do not corrupt core business logic.

<image src="../app/interface/static/flow.png" width=700>


#### Incremental Finite State Pipeline

Rather than relying on resource-intensive, one-shot prompt extractions, the system uses dual Qwen-3 Small Language Models (SLMs) to process conversational turns through an iterative state replay loop.

<image src="../app/interface/static/pipeline.png" width=700>


### 2.2 Component Topology Overview

At the operational tier, interaction vectors flow along the following sequence:

```text

 [ Browser Client ]
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ INTERFACE LAYER : FastAPI Web Engine (`app.interface.api`)                   │
 └──────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ CONTROL FAÇADE  : Handler & Validation Controllers (`app.interface.helpers`) │
 └──────┬───────────────────────────────┬───────────────────────────────┬───────┘
        │                               │                               │
        ▼ (Persistence)                 ▼ (Core Orchestration)          ▼ (Audio Stream)
 ┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │    CONCRETE DATA PROXIES    │ │       PROCESSING CORE       │ │      MULTIMODAL AUDIO       │
 │                             │ │                             │ │     PARSING SUBSYSTEMS      │
 │ (`app.infrastructure.       │ │ (`app.application.pipeline. │ │                             │
 │   persistence.mongo`)       │ │  FormFillingService`)       │ │  (ASR / Translation /       │
 └──────────────┬──────────────┘ └──────────────┬──────────────┘ │   Diarization)              │
                │                               │                └─────────────────────────────┘
                │                               ├───────────────────────────────┐
                │                               │                               │
                ▼                               ▼                               ▼
 ┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │       DATABASE TIER         │ │    PLUGGABLE EXTRACTION     │ │   ABSTRACTIVE SUMMARY       │
 │                             │ │           ENGINE            │ │           ENGINE            │
 │    MongoDB Cluster State    │ │                             │ │                             │
 └─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘

-----------------------------------------------------------------------------------
OPTIONAL MICROSERVICE DISTRIBUTED MODEL BOUNDARY:
-----------------------------------------------------------------------------------
 [ FastAPI Web Engine ] ──(HTTP Contract)──> [ Model Worker App ] ──> [ Local Transformer Weights ]
                                             (`app.interface.model_service_api`)

```

---

## 3. Layered Structure

The project implements a Clean Architecture directory structure to isolate software components:

```text
app/
├── domain/         # Domain state definitions and abstract interface contracts
├── application/    # Transaction orchestration rules and schema validation logic
├── infrastructure/ # Low-level file drivers, database proxies, and machine learning models
└── interface/      # Network transport interfaces, routing mechanics, and templates

```

### 3.1 `domain/`

This module acts as an isolated boundary containing the platform's core business requirements and data invariants. It has no external dependencies on frameworks, databases, or third-party client drivers.

* `domain.py`: Contains immutable data representations implemented via Pydantic dataclasses (e.g., entity definitions for logs, forms, and conversations).
* `interfaces.py`: Declares abstract behavioral signatures and structural boundaries via Python’s `abc.ABC` protocol definitions.
* `speakers.py`: Implements parsing mechanics for multi-speaker identification arrays to prevent collisions while mapping conversational history data streams.

### 3.2 `application/`

This module orchestrates business processes and controls the execution flows across the system use cases.

* `pipeline.py`: Houses the central data-filling workflow context (`FormFillingService`). It maps inputs into the target form schema, enforces data validation, tracking systems metrics, and commits final states to storage.

### 3.3 `infrastructure/`

This module contains the physical adapters required to link the application with external systems, databases, networks, and hardware acceleration drivers.

* `persistence/`: Implements explicit data access objects for MongoDB collections, managing persistence lifecycles and query performance profiles.
* `ai/`: Packages machine learning code into standard, swappable execution blocks for automatic speech recognition (ASR), translation, text extraction, and abstractive summarization.

### 3.4 `interface/`

This module defines user interaction boundaries and exposes the application's processing workflows over web networks.

* `api.py` / `helpers.py`: Manages high-concurrency web endpoints, authentication states, permission matrices, session parsing rules, and error tracking mechanisms.
* `collab_ws.py`: Implements an asynchronous WebSocket event coordinator to handle real-time collaboration sessions across shared workspaces.

---

## 4. Composition Root and Dependency Wiring

The orchestration center for dependency management resides in `app/interface/dependencies.py`. This module handles database connection bootstrapping, instantiates concrete data proxy drivers, and resolves the system's machine learning pipelines.

At startup, the runtime parses the environment configuration to select and initialize the active inference backend according to this prioritization chain:

$$\text{MOCK\_MODELS} \longrightarrow \text{USE\_MODAL\_INFERENCE} \longrightarrow \text{USE\_LOCAL\_CONTAINER\_GEMMA4} \longrightarrow \text{MODEL\_SERVICE\_URL} \longrightarrow \text{USE\_OLLAMA} \longrightarrow \text{Bundled Transformers}$$

This configuration strategy ensures that the underlying extraction pipelines can alternate between local mocks, cloud instances, container proxies, or direct local GPU inference arrays without modifying route handlers or application services.

---

## 5. Request and Processing Flows

### 5.1 Form Creation Flow

1. The operator issues an HTTP request to `/forms/new`.
2. `FormHandler.create_form` processes the request payload and builds a validated `FormSchema` object.
3. The schema is committed to persistent storage through the `MongoFormRepository` driver.

*Structural Detail:* Initial schemas store structured metadata and type constraints (e.g., `string`, `int`, `email`). During updates, fields accept raw natural-language validation instructions, letting the system use form properties as both schema type validators and contextual extraction instructions.

### 5.2 Static Text Extraction Flow

1. A structured conversation transcript is entered using clear speaker tags (`Speaker: Context`).
2. `ConversationParser.parse` splits the stream into a chronologically sorted internal timeline.
3. `ConversationHandler.create_conversation` logs the dialogue record into the application database.
4. `FormFillingService.run` loads the targeted conversation version and associated extraction schema.
5. The pipeline routes data payloads to the configured model backends to handle text extraction and summary generation tasks.
6. Parsed fields are validated, system execution log states are updated, and extracted data objects are committed to the outputs store.

### 5.3 Live Extraction Flow

Live interactions route requests directly through `_extract_for_conversation_text` inside `helpers.py`. This path processes dialogue turns incrementally, using streaming endpoints to continuously update state previews in the UI as the interaction unfolds. If supported by the underlying backend, the system parses structural information outside the predefined schema and surfaces real-time field suggestions to the operator.

### 5.4 Audio Extraction Flow

Multimedia uploads follow a multi-stage decoding pipeline:

1. The raw audio stream is transcribed by the processing tier.
2. If the language is not English, the transcript routes through a machine translation step before extraction.
3. Optional speaker diarization tasks isolate distinct acoustic channels and assign clear speaker tags across the timeline.
4. The processed text is normalized into standard structural turns and replayed sequentially through the core extraction state-machine logic.

### 5.5 Collaborative Conversation Flow

Real-time user synchronization is handled by the asynchronous room-manager defined in `collab_ws.py`. It manages multiplexed connections grouped by a distinct `room_id`, broadcasting typing statuses, user presences, and text buffer edits to all active session viewports. Persistent storage updates remain completely isolated from websocket operations; text changes are only saved when explicitly committed by a user handler.

---

## 6. Domain Models

Core business representations are managed as validated entities inside `app/domain/domain.py`:

### `ConversationVersion`

Models a frozen historical snapshot of a given conversation. This structure isolates edits across version updates to preserve the full audit history of the transaction.

* `version_index`: Integer tracking change history.
* `timestamp`: Real-world creation time.
* `history`: Parsed sequential dialogue data matrix.
* `run_id`: Execution run identifier tracking the associated processing pass.
* `source_mode` / `input_language`: Track original data configurations and ingestion states.
* `raw_transcript` / `translated_transcript`: Explicit logs tracking data states across the speech-to-text and machine translation pipelines.

### `Conversation`

Acts as the parent container tracking historical dialogue turns linked to a form. It groups all chronological `ConversationVersion` elements, exposing helper functions like `.latest_history` and `.full_text` to parse the underlying logs into clean text blocks for inference engines.

### `FormSchema`

Defines data structures and ownership boundaries for system forms. The schema property tracks an explicit flat key-value map (`field_name -> structural_instruction`) where values support both strict algorithmic type-checking constraints and natural-language extraction strings.

### `ExtractionResult`

Captures output data produced by a successful extraction pass. It maps extraction payloads (`filled_data`), dynamic out-of-schema field modifications (`accepted_new_fields`), tracking system keys (`run_id`), and high-density abstractive summaries.

### `ExtractionRequest`

An infrastructure data container used to manage per-field extractions in batch model configurations. It isolates context metadata, operational keys, field processing strings, and data type validation guidelines.

### `RunLog`

The audit ledger tracking application pipeline states. It captures execution durations (`started_at`, `finished_at`), pipeline statuses (`running`, `completed`, `failed`), error exceptions, text summaries, and extracted field snapshots.

---

## 7. Domain Interfaces

The contract specifications defined in `app/domain/interfaces.py` prevent infrastructure details from leaking into core application layers. Key abstractions include:

* `IFormRepository` / `IConversationRepository`: Enforce persistence signatures for structural components.
* `IExtractionModel` / `ISummarizer`: Standardize prediction signatures for downstream language models.
* `IPipeline` / `IRunLogRepository`: Expose workflow execution methods and auditing frameworks across the platform.

---

## 8. Persistence Design

Low-level operational queries targeting the MongoDB database are encapsulated inside `app/infrastructure/persistence/mongo.py`. The data persistence tier maps engine document boundaries across five distinct collections: `forms`, `conversations`, `run_logs`, `outputs`, and `users`.

Repository classes implement interface contracts to run CRUD statements cleanly through the `pymongo` driver. To avoid performance drop-offs as logs scale over time, the initialization layers ensure clear index mappings on critical fields, such as `run_id`, `started_at`, and the compound lookup key `(conversation_id, version_index)`.

---

## 9. Conversation Representation

The internal layout handles dialogue storage using a key-value format engineered to preserve interaction contexts:

*Input Dialogue String:*

```text
Doctor: Baseline assessment completed.
Patient: Confirming clarity.

```

*Internal Dictionary Layout:*

```python
{
  "Doctor 000001": "Baseline assessment completed.",
  "Patient 000002": "Confirming clarity.",
}

```

This map configuration delivers two primary engineering advantages: it maintains exact chronological insertion order while preventing dictionary key collision errors during extended interactions with repeated speakers. The speaker engine (`app/domain/speakers.py`) uses regex filtering to strip these numeric tracking suffixes before compiling text payloads for downstream inference engines.

---

## 10. Pipeline Design

The system processing core is managed by the `FormFillingService` class inside `app/application/pipeline.py`. When triggered, the service generates a distinct tracking run-id, logs an active processing record to the database, pulls relevant dialogue entities, and coordinates downstream model calls.

### 10.1 Type Normalization

The system validates engine outputs through `validate_field_type(...)`. This function enforces programmatic validation checks on data fields matching basic structural primitives: `string`, `int`, `float`, `email`, `phone`, and `date`. If an output field cannot be parsed into one of these strict signatures, the system skips validation and records the string as a raw fallback, enabling unstructured extractions from modified instruct templates.

### 10.2 Incremental Extraction Design

When processing a `full_process` request, the pipeline sets up form values as $N/A$ and breaks the transcript down turn by turn. It routes these entries into continuous calls to `process_live_update(...)`, carrying the accumulated state matrix forward across turns. Modeling extraction as an evolving state machine over sequential steps reduces memory overhead and improves output stability compared to heavy, context-wide batch extractions.

---

## 11. Extraction and Summarization Backends

The `app/infrastructure/ai/` directory packages deep learning code into standardized, pluggable files.

### 11.1 Extraction Backends

* `MockExtractionModel`: A low-overhead testing driver that skips machine learning steps to return mock states and placeholders.
* `RemoteModelServiceExtractionModel`: Renders text arrays over network links to hit an isolated microservice worker via an explicit HTTP contract.
* `ModalExtractionModel`: Handles task routing to serverless infrastructure pools via the distributed `modal` Python client SDK.
* `OllamaFormStateModel`: Interfaces with a local Ollama server, enforcing schema structure on model outputs by passing prompt templates that require clean JSON payloads containing keys for filled data and structural schema updates.
* `FormStateModel`: The system's primary local extraction driver. It instantiates local tokenizers and causal language models augmented with custom LoRA adapters, passing structured system logs to extract tool-style JSON state adjustments.
* `GemmaFunctionalModel` / `LocalHuggingFaceModel`: Fallback extraction paths that support legacy pipeline steps, including batch question-answering runs backed by validation confidence scores.

### 11.2 Summarization Backends

* `MockSummarizer`: Returns placeholder text to bypass summarization steps during interface testing.
* `RemoteModelServiceSummarizer` / `ModalSummarizer`: Offload heavy summarization jobs to remote microservice endpoints or cloud instances.
* `OllamaSummarizer`: Connects to local containerized setups to run prompt templates designed to condense long text records.
* `LocalSummarizer`: Runs an isolated sequence-to-sequence model (`sshleifer/distilbart-cnn-12-6`) to compress tracking logs locally.
* `GemmaSummarizer` / `QwenSummarizer`: Generate text summaries via causal instruct language models using iterative, turn-by-turn context adjustments or comprehensive one-shot prompts.

---

## 12. Audio and Language Helpers

Speech processing tasks are isolated within specialized infrastructure components:

* `LocalASRTranscriber`: Converts raw audio data into text tokens using an uncompressed `openai/whisper-small` network model.
* `LocalSpeechToText`: An alternative processing path for real-time mic previews. It uses automated voice activity detection (VAD), clears up low-frequency background noise, runs the Google SpeechRecognition backend, and formats written numeric tokens back into plain integer digits.
* `LocalSpeakerDiarizer`: Separates compound audio channels into distinct speaker segments. It pairs Whisper time logs with PyAnnote acoustic embeddings, runs agglomerative clustering to identify independent speakers, and groups sequential turns into clean text logs.
* `LocalTranslator`: Automatically translates text strings from recognized foreign tongues (Spanish, French, German, Italian, and Portuguese) into English targets before extraction.

---

## 13. Interface Layer Design

The web interface separates network configuration, validation code, data formats, and asynchronous script tasks.

* `api.py`: Configures the main FastAPI application context, maps folder asset tracks, hooks exception middleware routines, and initializes lifespans to securely spin up or tear down database connections.
* `helpers.py`: Acts as a comprehensive facade layer for web routes. It groups authentication checks, routes user access queries, parses unstructured text fields, manages session merges, and orchestrates transaction handlers across the interface tier.

---

## 14. Access Control Model

The system enforces data security boundaries across endpoints via the `AccessPolicy` security guard.

```text
               +--------------------------------------+
               |      Request Context Evaluation      |
               +------------------+-------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
  [ User Identity ]                                 [ Admin Identity ]
         |                                                 |
         v                                                 v
+-----------------------------+               +----------------------------+
| Is Personal Object Owner?   |               | Superuser Bypass           |
| Is Explicit Collaborator?   |               | Full Global Read/Write     |
| Is Global System Template?  |               +----------------------------+
+-----------------------------+

```

This security engine evaluates incoming identities against three distinct operational rules: Administrators bypass constraints with full global read and write privileges; forms designated as global remain visible to all authenticated users; and modifications to private conversations remain locked to the validated owner account or explicit session collaborators.

---

## 15. Model-Service Subsystem

`app/interface/model_service_api.py` exposes a secondary, lightweight FastAPI server designed to package and isolate heavy machine learning tasks. It exposes core utility routes over a predictable network interface:

* `GET /health`: Tracking service availability.
* `POST /extract` / `POST /live-extract`: Direct targets for text extraction passes.
* `POST /summarize`: Orchestrates text abstraction routines.

Isolating high-compute model runtimes within an independent worker container keeps the user-facing web process highly responsive and allows the application tier to scale independently.

---

## 16. Configuration Management

Application settings are parsed from environment variables into structured configuration objects via `Pydantic-Settings` inside `app/infrastructure/config.py`. This structure enforces validation on critical runtime values, including connection targets (`MONGO_URI`), microservice boundaries (`MODEL_SERVICE_URL`), feature flags (`MOCK_MODELS`), and model weight configurations (`FORM_STATE_MODEL_PATH`, `SUMMARIZER_MODEL_PATH`).

---

## 17. Testing Structure

The verification suite contains two isolated operational testing frameworks:

1. **Integration Core (`tests/test_interface_flows.py`):** Runs automated route validation checks using mock backends, verifying account registration flows, schema creation adjustments, and state data merges.
2. **End-to-End Suite (`tests/e2e/test_ui_flows.py`):** Uses the Playwright browser automation framework to simulate end-to-end user interactions, testing dashboard navigation, dynamic form rendering, typed text ingestion, and systemic audit logs.

---

## 18. Core Architectural Tradeoffs & Caveats

1. **Sequential State Replay vs. One-Shot Extraction:**
Running extractions turn-by-turn processes long text tracks linearly ($O(N)$ prompt requests), which introduces higher latency compared to a single comprehensive batch call. However, this incremental state approach matches live streaming data pipelines and prevents context hallucinations over extended conversations.
2. **Shared-Duty Properties:**
Using schema objects to track both strict validation datatypes and loose text extraction guidelines minimizes codebase duplication. The tradeoff is that adjustments to prompt text require careful coordination to avoid impacting downstream programmatic validation logic.
3. **Silent Error Callbacks:**
When remote model layers fail or become unreachable, the system automatically degrades to mock execution wrappers. While this fallback strategy prevents user-facing errors, it can mask internal infrastructure misconfigurations unless monitoring tools are closely attached to system logs.