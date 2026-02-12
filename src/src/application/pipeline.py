from uuid import uuid4
from datetime import datetime
import json
from ..domain.domain import ExtractionResult, ExtractionRequest, RunLog
from ..domain.interfaces import IConversationRepository, IFormRepository, IExtractionModel, IPipeline, IRunLogRepository

class FormFillingService(IPipeline):
    def __init__(self, conversation_repo: IConversationRepository, form_repo: IFormRepository, extraction_model: IExtractionModel, runlog_repo: IRunLogRepository, model_type="field_process"): # field_process models process one field after onother. full_process processes all fields at once.
        self.convo_repo = conversation_repo
        self.form_repo = form_repo
        self.model = extraction_model
        self.runlog_repo = runlog_repo
        self.model_type = model_type

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
            requests = []
            field_keys = []

            empty_fields = {}

            full_convo = ""

            for speaker, text in convo.latest_history.items():
                if " " in speaker:
                    speaker = " ".join(speaker.split()[:-1])
                full_convo += speaker + ": " + text + "\n"

            for field_key, question in form.fields.items():
                field_keys.append(field_key)
                if self.model_type == "field_process":
                    req = ExtractionRequest(
                        context=context,
                        field_name=field_key,
                        instruction=question
                    )
                    requests.append(req)
                    
                elif self.model_type == "full_process":
                    empty_fields[field_key] = "N/A"
                else:
                    raise Exception

            if self.model_type == "field_process":
                # 3. Run Batch Extraction (Solves N+1 problem)
                answers = await self.model.extract_batch(requests)
            elif self.model_type == "full_process":
                input_str = f"""Extract info from conversation to fill form.\nConversation: {full_convo}Form: {form.name}\nFields: {json.dumps(empty_fields)}"""
                answers = await self.model.process_extraction_request(input_str)
                print(answers)
            else:
                raise Exception 

            # 4. Compile Results
            filled_data = {}
            for key, value in zip(field_keys, answers):
                print(key, value)
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
            print(filled_data)
            result = ExtractionResult(
                conversation_id=conversation_id,
                form_id=form_id,
                filled_data=filled_data,
                run_id=run_id
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

