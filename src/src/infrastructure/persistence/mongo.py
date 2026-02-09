import motor.motor_asyncio
from typing import List, Optional
from ...domain.interfaces import IConversationRepository, IFormRepository, IRunLogRepository
from ...domain.domain import Conversation, FormSchema, RunLog

class MongoConversationRepository(IConversationRepository):
    def __init__(self, connection_string: str, db_name: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db.conversations

    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        doc = await self.collection.find_one({"conversation_id": conversation_id})
        return Conversation(**doc) if doc else None

    async def get_by_form_id(self, form_id: str) -> List[Conversation]:
        cursor = self.collection.find({"form_id": form_id})
        docs = await cursor.to_list(length=None)
        return [Conversation(**doc) for doc in docs]

    async def save(self, conversation: Conversation) -> None:
        if not conversation.versions:
            return
            
        await self.collection.update_one(
            {"conversation_id": conversation.id},
            {"$set": conversation.model_dump(by_alias=True)},
            upsert=True
        )

class MongoFormRepository(IFormRepository):
    def __init__(self, connection_string: str, db_name: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db.forms

    async def get_all(self) -> List[FormSchema]:
        cursor = self.collection.find({})
        docs = await cursor.to_list(length=None)
        return [FormSchema(**doc) for doc in docs]

    async def get_by_id(self, form_id: str) -> Optional[FormSchema]:
        doc = await self.collection.find_one({"form_id": form_id})
        return FormSchema(**doc) if doc else None
    
    async def save(self, form: FormSchema) -> None:
        await self.collection.update_one(
            {"form_id": form.id},
            {"$set": form.model_dump(by_alias=True)},
            upsert=True
        )

class MongoRunLogRepo(IRunLogRepository):
    def __init__(self, connection_string: str, db_name: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db.run_logs

    async def ensure_indexes(self):
        await self.collection.create_index("run_id", unique=True)
        await self.collection.create_index("started_at")

    async def create(self, log: RunLog) -> None:
        await self.collection.insert_one(log.model_dump(by_alias=True))

    async def update(self, run_id: str, data: dict) -> None:
        await self.collection.update_one(
            {"run_id": run_id},
            {"$set": data}
        )

    async def get_recent(self, limit: int = 20) -> List[RunLog]:
        cursor = self.collection.find().sort("started_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        cleaned = []
        for doc in docs:
            doc.pop("_id", None)   
            cleaned.append(RunLog(**doc))

        return cleaned

    async def get_by_id(self, run_id: str) -> Optional[RunLog]:
        doc = await self.collection.find_one({"run_id": run_id})
        if doc:
            doc.pop(" _id", None)
            return RunLog(**doc) 
        else:
            return None

