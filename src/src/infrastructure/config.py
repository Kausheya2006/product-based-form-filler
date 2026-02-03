from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://mongodb:27017"
    DB_NAME: str = "chat_db"
    MODEL_NAME: str = "distilbert-base-cased-distilled-squad"

    class Config:
        env_file = ".env"

settings = Settings()
