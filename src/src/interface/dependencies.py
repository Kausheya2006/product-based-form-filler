"""Dependency Injection - Composition Root"""
import logging
from ..domain.interfaces import IConversationRepository, IFormRepository, IPipeline, IRunLogRepository, ISummarizer
from ..infrastructure.persistence.mongo import MongoConversationRepository, MongoFormRepository, MongoRunLogRepo
from ..infrastructure.ai.local_model import LocalHuggingFaceModel, GemmaFunctionalModel, FormStateModel
from ..infrastructure.ai.summarizer import LocalSummarizer, GemmaSummarizer, QwenSummarizer
from ..infrastructure.ai.ollama_model import OllamaFormStateModel, OllamaSummarizer
from ..infrastructure.ai.model_service_client import RemoteModelServiceExtractionModel, RemoteModelServiceSummarizer
from ..infrastructure.ai.mock_models import MockExtractionModel, MockSummarizer
from ..infrastructure.ai.translator import LocalTranslator
from ..infrastructure.ai.asr import LocalASRTranscriber
from ..infrastructure.ai.stt import LocalSpeechToText
from ..infrastructure.config import settings
from ..application.pipeline import FormFillingService

logger = logging.getLogger(__name__)

class Container:
    """DI Container - wires interfaces to implementations"""
    convo_repo: IConversationRepository = None
    form_repo: IFormRepository = None
    runlog_repo: IRunLogRepository = None 
    pipeline: IPipeline = None
    translator: LocalTranslator = None
    asr_transcriber: LocalASRTranscriber = None
    stt_service: LocalSpeechToText = None

    @classmethod
    def initialize(cls):
        cls.convo_repo = MongoConversationRepository(settings.MONGO_URI, settings.DB_NAME)
        cls.form_repo = MongoFormRepository(settings.MONGO_URI, settings.DB_NAME)
        cls.runlog_repo = MongoRunLogRepo(settings.MONGO_URI, settings.DB_NAME)
        if settings.MOCK_MODELS:
            logger.info("MOCK_MODELS=true — skipping ML model loading (no GPU required)")
            model = MockExtractionModel()
            summarizer = MockSummarizer()
        elif settings.MODEL_SERVICE_URL:
            logger.info(
                "MODEL_SERVICE_URL set — routing extraction/summarization to model service (%s)",
                settings.MODEL_SERVICE_URL,
            )
            model = RemoteModelServiceExtractionModel(settings.MODEL_SERVICE_URL)
            summarizer = RemoteModelServiceSummarizer(settings.MODEL_SERVICE_URL)
        elif settings.USE_OLLAMA:
            logger.info(
                "USE_OLLAMA=true — routing extraction/summarization to Ollama (%s)",
                settings.OLLAMA_BASE_URL,
            )
            model = OllamaFormStateModel(
                model_name=settings.OLLAMA_EXTRACT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
            )
            summarizer = OllamaSummarizer(
                model_name=settings.OLLAMA_SUMMARIZER_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
            )
        else:
            if settings.EXTRACTION_MODEL_TYPE == "form_state":
                model = FormStateModel(model_path=settings.FORM_STATE_MODEL_PATH)
            else:
                model = GemmaFunctionalModel(max_input_tokens=512, max_new_tokens=256, temperature=0.0, checkpoint_path="/app/data_generation/models/checkpoint-200")

            if settings.SUMMARIZER_TYPE == "gemma":
                summarizer = GemmaSummarizer(model_path=settings.SUMMARIZER_MODEL_PATH)
            elif settings.SUMMARIZER_TYPE == "qwen":
                summarizer = QwenSummarizer(model_name=settings.SUMMARIZER_MODEL_PATH)
            else:
                summarizer = LocalSummarizer()

        cls.pipeline = FormFillingService(cls.convo_repo, cls.form_repo, model, cls.runlog_repo, summarizer, model_type="full_process")
        cls.translator = LocalTranslator()
        cls.asr_transcriber = LocalASRTranscriber()
        cls.stt_service = LocalSpeechToText()

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
