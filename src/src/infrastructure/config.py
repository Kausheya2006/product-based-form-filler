from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb+srv://nehaprabhup17_db_user:welcome12@dass.rt4ytmt.mongodb.net/chat_db?retryWrites=true&w=majority"
    DB_NAME: str = "chat_db"
    MODEL_NAME: str = "distilbert-base-cased-distilled-squad"
    EXTRACTION_MODEL_TYPE: str = "form_state"  # "gemma_functional" or "form_state"
    FORM_STATE_MODEL_PATH: str = "/app/data_generation/monomodel/model"   # Currently Qwen2.5-1.5b with LoRA
    SUMMARIZER_TYPE: str = "qwen"  # "gemma", "qwen", or "distilbart"
    SUMMARIZER_MODEL_PATH: str = "Qwen/Qwen2.5-1.5B-Instruct"
    USE_OLLAMA: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EXTRACT_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_SUMMARIZER_MODEL: str = "qwen2.5:1.5b"
    MODEL_SERVICE_URL: str = ""
    ADMIN_USERNAME: str = "PLadmin"
    MOCK_MODELS: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
