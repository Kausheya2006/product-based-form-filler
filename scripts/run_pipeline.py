"""
How to use?
-> Change line 74 as per required number of inputs to test on
"""

import sys, os, json, asyncio
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(_file_), ".."))

from src.application.pipeline import FormFillingService
from src.domain.domain import Conversation, ConversationVersion, FormSchema
from src.domain.interfaces import IConversationRepository, IFormRepository, IRunLogRepository
from src.infrastructure.ai.local_model import GemmaFunctionalModel
from src.infrastructure.ai.summarizer import LocalSummarizer

DATA_DIR = os.path.join(os.path.dirname(_file_), "..", "data")

with open(os.path.join(DATA_DIR, "generated_conversations.json")) as f:
    raw_convs = json.load(f)
with open(os.path.join(DATA_DIR, "generated_forms.json")) as f:
    raw_forms = json.load(f)

forms_by_id = {f["form_id"]: f for f in raw_forms}


class InMemoryConvoRepo(IConversationRepository):
    def _init_(self, convs):
        self._convs = {c.id: c for c in convs}
    async def get_by_id(self, cid): return self._convs.get(cid)
    async def get_by_form_id(self, fid): return [c for c in self._convs.values() if c.form_id == fid]
    async def save(self, c): pass


class InMemoryFormRepo(IFormRepository):
    def _init_(self, forms):
        self._forms = {f.id: f for f in forms}
    async def get_all(self): return list(self._forms.values())
    async def get_by_id(self, fid): return self._forms.get(fid)
    async def save(self, f): pass
    async def delete_by_id(self, fid): pass


class NoOpRunLogRepo(IRunLogRepository):
    async def create(self, log): pass
    async def update(self, run_id, data): pass
    async def get_recent(self, limit=20): return []
    async def get_by_id(self, run_id): return None


def build_conversation(raw):
    versions = []
    for v in raw.get("versions", []):
        ts = v.get("timestamp")
        if isinstance(ts, dict) and "$date" in ts:
            ts = datetime.fromisoformat(ts["$date"].replace("Z", "+00:00"))
        elif isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            ts = datetime.utcnow()
        versions.append(ConversationVersion(
            version_index=v["version_index"],
            timestamp=ts,
            history=v["history"],
            run_id=v.get("run_id"),
        ))
    if not versions:
        versions = [ConversationVersion(version_index=0, history=raw["conversation"])]
    return Conversation(conversation_id=raw["conversation_id"], form_id=raw["form_id"], versions=versions)


async def main():
    pairs = []
    for raw in raw_convs[-110:-105]: # CHANGE THIS AS PER NEED
        fid = raw["form_id"]
        if fid not in forms_by_id:
            print(f"Warning: form {fid} not found, skipping")
            continue
        conv = build_conversation(raw)
        form = FormSchema.model_validate(forms_by_id[fid])
        pairs.append((conv, form))

    convo_repo = InMemoryConvoRepo([p[0] for p in pairs])
    form_repo = InMemoryFormRepo([p[1] for p in pairs])
    runlog_repo = NoOpRunLogRepo()
    model = GemmaFunctionalModel(max_input_tokens=8192, max_new_tokens=512, temperature=0.0, checkpoint_path="data_generation/models/checkpoint-200")
    summarizer = LocalSummarizer()
    pipeline = FormFillingService(convo_repo, form_repo, model, runlog_repo, summarizer, model_type="full_process")

    results = []
    for i, (conv, form) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] conv={conv.id} form={form.id}")
        try:
            result = await pipeline.run(conv.id, form.id, 0)
            results.append({
                "conversation_id": result.conversation_id,
                "form_id": result.form_id,
                "filled_data": result.filled_data,
                "summary": result.summary,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"conversation_id": conv.id, "form_id": form.id, "error": str(e)})

    output_path = "output.json"
    with open(output_path, "a") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)} results -> {output_path}")


asyncio.run(main())