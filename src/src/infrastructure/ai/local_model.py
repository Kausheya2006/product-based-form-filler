import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from pathlib import Path
from transformers import pipeline
from ...domain.interfaces import IExtractionModel
from ...domain.domain import ExtractionRequest
from ..config import settings

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # system_prompt = "You are retrieving information from the above conversation to retrieve the data of a particular field. Answer N/A if the information is not available. Retrieve the information which matches this given field name the closest: "
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

class GemmaFunctionalModel(IExtractionModel):
    def __init__(self, max_input_tokens=512, max_new_tokens=256, temperature=0.0, checkpoint_path="data_generation/models/checkpoint-200"):
        print(f"Loading local model: Gemma Functional ")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        resolved_checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_checkpoint_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        candidate = Path(checkpoint_path).expanduser()
        file_dir = Path(__file__).resolve()
        app_root = file_dir.parents[3]

        candidates = [
            candidate,
            Path.cwd() / candidate,
            app_root / candidate,
            app_root / "data_generation/models/checkpoint-200",
            Path("/app/data_generation/models/checkpoint-200"),
            Path("/app/src/data_generation/models/checkpoint-200"),
        ]

        for path in candidates:
            if path.is_dir() and (path / "config.json").is_file():
                return str(path)

        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Checkpoint directory not found or incomplete for '{checkpoint_path}'. "
            f"Tried: {checked}"
        )

    async def process_extraction_request(self, input_str: str) -> List[Any]:
        full_input = f"<start_of_turn>user\n{input_str}<end_of_turn>\n<start_of_turn>model\n"

        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        def parse_model_json(text):
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        prediction = parse_model_json(generated)

        return list(prediction.values())
