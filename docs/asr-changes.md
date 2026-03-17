# Architecture & Design Changes Summary (ASR + Translation Flow)

## Scope
This update adds a new **audio-first extraction path** while keeping the existing static/live text flows unchanged.

## 1) New service boundaries in Infrastructure
- Added `LocalASRTranscriber` in `src/src/infrastructure/ai/asr.py`.
  - Responsibility: convert uploaded/recorded audio into transcript text.
- Added `LocalTranslator` in `src/src/infrastructure/ai/translator.py`.
  - Responsibility: translate non-English transcript text to English.

## 2) Composition root / DI changes
- `Container` now wires two new adapters:
  - `translator`
  - `asr_transcriber`
- This preserves Clean Architecture direction: interface routes depend on abstractions/composed services, not on model internals.

## 3) Interface/API flow changes
- Added ASR page route: `GET /forms/{form_id}/asr`.
- Extended ASR create route: `POST /conversations/create-asr` to accept `audio_file`.
- Processing order in ASR route:
  1. Audio upload/recording payload
  2. Transcription
  3. Translation to English
  4. Existing static extraction persistence helper
- Reused shared helper `_persist_conversation_and_extract(...)` to avoid duplicating extraction/persistence logic.

## 4) Domain model evolution
- `ConversationVersion` now includes source metadata:
  - `source_mode` (`text` or `asr`)
  - `input_language`
  - `raw_transcript`
  - `translated_transcript`
- Design intent: ASR-generated records are distinguishable/auditable from normal text-entered conversations.

## 4.1) Save behavior (ASR vs text)
- Both flows still save into the same `conversations` collection/repository path and use the shared `_persist_conversation_and_extract(...)` helper.
- The **difference** is in `versions[0]` metadata:
  - Text flow writes `source_mode: "text"`.
  - ASR flow writes `source_mode: "asr"` plus `input_language`, `raw_transcript`, and `translated_transcript`.
- ASR transcript is normalized to parser-compatible history as `Speaker: <translated text>` before extraction, so extraction remains compatible with the existing static pipeline.

## 5) UI design changes
- Added a fourth action entry point on form detail: **ASR Extraction**.
- New ASR page supports:
  - language selection
  - audio file upload
  - in-browser recording (start/stop + preview)
- Form submits multipart data to the ASR route.

## 6) Runtime dependency updates
- Added audio stack dependencies in `src/requirements.txt`:
  - `librosa`
  - `soundfile`

## Net design outcome
The system now supports a **Audio -> Transcript -> English -> Static Extraction** pipeline
