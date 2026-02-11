import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def format_sample(item):
    fields_empty = {k: "N/A" for k in item['fields'].keys()}
    inp = f"""Extract info from conversation to fill form.

Conversation: {item['conversation']}
Form: {item['form_name']}
Fields: {json.dumps(fields_empty)}"""
    out = json.dumps(item['fields'])
    return {"text": f"<start_of_turn>user\n{inp}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn>", "input": inp, "output": out}

def prepare_data(filepath):
    data = json.load(open(filepath))
    formatted = [format_sample(item) for item in data]
    split = int(0.9 * len(formatted))
    return Dataset.from_list(formatted[:split]), Dataset.from_list(formatted[split:])

def train_model(model_name, train_ds, eval_ds, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_steps=50,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        packing=False,
        report_to="none"
    )
    
    trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

def evaluate(model, tokenizer, test_ds):
    model.eval()
    correct, total, exact = 0, 0, 0
    
    for example in test_ds:
        expected = json.loads(example['output'])
        inputs = tokenizer(f"<start_of_turn>user\n{example['input']}<end_of_turn>\n<start_of_turn>model\n", return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        try:
            predicted = json.loads(generated[generated.find("{"):generated.rfind("}")+1]) if "{" in generated else {}
        except:
            predicted = {}
        
        exact += (predicted == expected)
        for k in expected:
            total += 1
            correct += (k in predicted and predicted[k] == expected[k])
    
    return {"field_acc": correct/total*100 if total else 0, "exact_acc": exact/len(test_ds)*100 if test_ds else 0}

if __name__ == "__main__":
    train_ds, test_ds = prepare_data("data/training_data.json")
    model, tokenizer = train_model("google/functiongemma-270m-it", train_ds, test_ds, "data_generation/models")
    results = evaluate(model, tokenizer, test_ds)
    json.dump(results, open("data_generation/models/results.json", 'w'), indent=2)
