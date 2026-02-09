from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb+srv://nehaprabhup17_db_user:welcome12@dass.rt4ytmt.mongodb.net/chat_db?retryWrites=true&w=majority"
    DB_NAME: str = "chat_db"
    MODEL_NAME: str = "distilbert-base-cased-distilled-squad"

    class Config:
        env_file = ".env"

settings = Settings()
