from abc import ABC, abstractmethod
from typing import List, Any, Optional
from .domain import Conversation, FormSchema, ExtractionResult, ExtractionRequest, RunLog

class IConversationRepository(ABC):
    @abstractmethod
    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]: pass
    @abstractmethod
    async def get_by_form_id(self, form_id: str) -> List[Conversation]: pass
    @abstractmethod
    async def save(self, conversation: Conversation) -> None: pass

class IFormRepository(ABC):
    @abstractmethod
    async def get_all(self) -> List[FormSchema]: pass
    @abstractmethod
    async def get_by_id(self, form_id: str) -> Optional[FormSchema]: pass
    @abstractmethod
    async def save(self, form: FormSchema) -> None: pass

class IExtractionModel(ABC):
    @abstractmethod
    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]: pass

class IPipeline(ABC):
    @abstractmethod
    async def run(self, conversation_id: str, form_id: str) -> ExtractionResult: pass

class IRunLogRepository(ABC): 
    @abstractmethod
    async def create(self, log: RunLog) -> None: pass
    @abstractmethod
    async def update(self, run_id: str, data: dict) -> None: pass
    @abstractmethod
    async def get_recent(self, limit: int = 20) -> List[RunLog]: pass
    @abstractmethod
    async def get_by_id(self, run_id: str) -> Optional[RunLog]: pass


