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
