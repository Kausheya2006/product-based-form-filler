"""Dependency Injection - Composition Root"""
import logging
from ..domain.interfaces import IConversationRepository, IFormRepository, IPipeline, IRunLogRepository, ISummarizer
from ..infrastructure.persistence.mongo import MongoConversationRepository, MongoFormRepository, MongoRunLogRepo
from ..infrastructure.ai.local_model import LocalHuggingFaceModel, GemmaFunctionalModel
from ..infrastructure.ai.summarizer import LocalSummarizer
from ..infrastructure.config import settings
from ..application.pipeline import FormFillingService

logger = logging.getLogger(__name__)

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
        model = GemmaFunctionalModel(max_input_tokens=512, max_new_tokens=256, temperature=0.0, checkpoint_path="/app/data_generation/models/checkpoint-200")
        summarizer = LocalSummarizer()
        cls.pipeline = FormFillingService(cls.convo_repo, cls.form_repo, model, cls.runlog_repo, summarizer, model_type="full_process")

        # Log which Mongo host is being used (mask credentials)
        uri = settings.MONGO_URI
        masked = uri
        try:
            scheme, rest = uri.split("://", 1)
            if "@" in rest:
                userinfo, host = rest.split("@", 1)
                masked = f"{scheme}://***@{host}"
        except Exception:
            masked = uri
        logger.info(f"Mongo URI in use: {masked}")

container = Container()
