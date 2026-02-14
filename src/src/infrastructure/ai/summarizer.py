from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ...domain.interfaces import ISummarizer

class LocalSummarizer(ISummarizer):
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        # Load components manually
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    async def summarize(self, text: str) -> str:
        # Tokenize input with truncation (model has 1024 token limit)
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and return
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary