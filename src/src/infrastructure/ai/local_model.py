import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from transformers import pipeline
from ...domain.interfaces import IExtractionModel
from ...domain.domain import ExtractionRequest
from ..config import settings

class LocalHuggingFaceModel(IExtractionModel):
    def __init__(self):
        model_name = settings.MODEL_NAME
        print(f"Loading local model: {model_name}...")
        self.pipeline = pipeline("question-answering", model=model_name)
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        """
        Extracts multiple fields in one go. 
        HuggingFace pipelines are optimized for batching.
        """
        if not requests:
            return []

        # Prepare inputs for the pipeline
        # Passing separate lists for questions and contexts to avoid DeprecationWarning
        questions = [req.instruction for req in requests]
        contexts = [req.context for req in requests]
        
        loop = asyncio.get_running_loop()
        try:
            # We run the batch inference call in a separate thread
            # pipeline(question=..., context=...) handles batching natively
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.pipeline(question=questions, context=contexts)
            )
            # results will be [{'score':.., 'start':.., 'end':.., 'answer':..}, ...]
            return [r['answer'] if r else None for r in results]
        except Exception as e:
            print(f"Batch extraction error: {e}")
            return [None] * len(requests)
