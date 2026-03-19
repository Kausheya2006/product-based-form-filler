from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb+srv://nehaprabhup17_db_user:welcome12@dass.rt4ytmt.mongodb.net/chat_db?retryWrites=true&w=majority"
    DB_NAME: str = "chat_db"
    MODEL_NAME: str = "distilbert-base-cased-distilled-squad"
    EXTRACTION_MODEL_TYPE: str = "gemma_form_state"  # "gemma_functional" or "gemma_form_state"
    FORM_STATE_MODEL_PATH: str = "/app/data_generation/monomodel/model"
    SUMMARIZER_TYPE: str = "qwen"  # "gemma", "qwen", or "distilbart"
    SUMMARIZER_MODEL_PATH: str = "Qwen/Qwen2.5-1.5B-Instruct"

    class Config:
        env_file = ".env"

settings = Settings()
