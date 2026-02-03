"""Dependency Injection - Composition Root"""
from ..domain.interfaces import IConversationRepository, IFormRepository, IPipeline, IRunLogRepository
from ..infrastructure.persistence.mongo import MongoConversationRepository, MongoFormRepository, MongoRunLogRepo
from ..infrastructure.ai.local_model import LocalHuggingFaceModel
from ..infrastructure.config import settings
from ..application.pipeline import FormFillingService

class Container:
    """DI Container - wires interfaces to implementations"""
    convo_repo: IConversationRepository = None
    form_repo: IFormRepository = None
    runlog_repo: IRunLogRepository = None 
    pipeline: IPipeline = None

    @classmethod
    def initialize(cls):
        cls.convo_repo = MongoConversationRepository(settings.MONGO_URI, settings.DB_NAME)
        cls.form_repo = MongoFormRepository(settings.MONGO_URI, settings.DB_NAME)
        cls.runlog_repo = MongoRunLogRepo(settings.MONGO_URI, settings.DB_NAME)
        model = LocalHuggingFaceModel()
        cls.pipeline = FormFillingService(cls.convo_repo, cls.form_repo, model, cls.runlog_repo)

container = Container()
