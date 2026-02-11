from uuid import uuid4
from datetime import datetime
from ..domain.domain import ExtractionResult, ExtractionRequest, RunLog
from ..domain.interfaces import IConversationRepository, IFormRepository, IExtractionModel, IPipeline, IRunLogRepository
from ..infrastructure.ai.local_model import smart_turn_chunks_generic

class FormFillingService(IPipeline):
    def __init__(self, conversation_repo: IConversationRepository, form_repo: IFormRepository, extraction_model: IExtractionModel, runlog_repo: IRunLogRepository, ):
        self.convo_repo = conversation_repo
        self.form_repo = form_repo
        self.model = extraction_model
        self.runlog_repo = runlog_repo

    async def run(self, conversation_id: str, form_id: str, version_index: int) -> ExtractionResult:
        run_id = str(uuid4())

        await self.runlog_repo.create(RunLog(
            run_id=run_id,
            conversation_id=conversation_id,
            version_index=version_index,
            started_at=datetime.utcnow(),
            status="running"
        ))

        try:
            # 1. Fetch Data
            convo = await self.convo_repo.get_by_id(conversation_id)
            if not convo:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            form = await self.form_repo.get_by_id(form_id)
            if not form:
                raise ValueError(f"Form {form_id} not found")

            # 2. Prepare Context & Batch Requests
            context = convo.full_text
            
            # Generate chunks for visibility (word-based chunking)
            chunks = smart_turn_chunks_generic(context, max_words=30)
            
            requests = []
            field_keys = []

            for field_key, question in form.fields.items():
                req = ExtractionRequest(
                    context=context,
                    field_name=field_key,
                    instruction=question
                )
                requests.append(req)
                field_keys.append(field_key)

            # 3. Run Batch Extraction (Solves N+1 problem)
            answers = await self.model.extract_batch(requests)

            # 4. Compile Results
            filled_data = {}
            for key, value in zip(field_keys, answers):
                # Handle nested keys like "address.street"
                if "." in key:
                    parts = key.split(".")
                    current = filled_data
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    filled_data[key] = value

            result = ExtractionResult(
                conversation_id=conversation_id,
                form_id=form_id,
                filled_data=filled_data,
                run_id=run_id,
                chunks=chunks
            )

            await self.runlog_repo.update(run_id, {
                "finished_at": datetime.utcnow(),
                "status": "success",
                "extracted_fields": result.model_dump()
            })

            return result
        except Exception as e:
            await self.runlog_repo.update(run_id, {
                "finished_at": datetime.utcnow(),
                "status": "failed",
                "error": str(e)
            })
            raise e

