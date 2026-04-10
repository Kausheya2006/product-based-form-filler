```sh
ollama serve # to start ollama

ollama pull qwen2.5:1.5b # only once (to download model)

export MONGO_URI="mongodb://localhost:27017/chat_db"
export DB_NAME="chat_db"
export MOCK_MODELS="false"
export USE_OLLAMA="true" 
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EXTRACT_MODEL="qwen2.5:1.5b"
export OLLAMA_SUMMARIZER_MODEL="qwen2.5:1.5b"

uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --reload
```