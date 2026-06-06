from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    DB_NAME: str
    MODEL_NAME: str = "distilbert-base-cased-distilled-squad"
    EXTRACTION_MODEL_TYPE: str = "form_state"  # "gemma_functional" or "form_state"
    FORM_STATE_MODEL_PATH: str = "/app/data_generation/monomodel/model"   # Currently Qwen2.5-1.5b with LoRA
    SUMMARIZER_TYPE: str = "qwen"  # "gemma", "qwen", or "distilbart"
    SUMMARIZER_MODEL_PATH: str = "Qwen/Qwen2.5-1.5B-Instruct"
    USE_OLLAMA: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EXTRACT_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_SUMMARIZER_MODEL: str = "qwen2.5:1.5b"
    USE_MODAL_INFERENCE: bool = False
    MODAL_INFERENCE_USE_SDK: bool = True
    MODAL_INFERENCE_URL: str = ""
    MODAL_APP_NAME: str = "monomodel-qwen3-4b-infer"
    MODAL_EXTRACT_FUNCTION: str = "modal_live_extract"
    MODAL_SUMMARIZER_FUNCTION: str = "modal_summarize"
    USE_LOCAL_CONTAINER_GEMMA4: bool = False
    LOCAL_CONTAINER_BASE_URL: str = "http://localhost:11434"
    LOCAL_CONTAINER_EXTRACT_MODEL: str = "gemma4-e2b:latest"
    LOCAL_CONTAINER_SUMMARIZER_MODEL: str = "gemma4-e2b:latest"
    MODEL_SERVICE_URL: str = ""
    ADMIN_USERNAME: str = "PLadmin"
    MOCK_MODELS: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
