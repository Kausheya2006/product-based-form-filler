import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Paths
base_path = "/home/orbi8r/storage/prog/DASS/project-monorepo/project-monorepo-team-27/src/data_generation/monomodel"
model_id = os.path.join(base_path, "base_model")
dataset_path = os.path.join(base_path, "curated_datapoints.json")
output_dir = os.path.join(base_path, "checkpoints")
final_model_dir = os.path.join(base_path, "model")

# 1. Load Dataset
with open(dataset_path, "r") as f:
    data = json.load(f)

# Take first 400
data = data[:400]
print(f"Loaded {len(data)} datapoints for training & validation.")

# Extract only the 'text' field to avoid pyarrow type errors
formatted_data = [{"text": d["text"]} for d in data]

# Split into train (90%) and eval (10%)
train_size = int(0.9 * len(formatted_data))
train_data = formatted_data[:train_size]
eval_data = formatted_data[train_size:]

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# 2. Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # GTX 1650 needs fp16, NOT bf16
)

# 3. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 4. Prepare for PEFT
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False  # silence the warnings

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8, # Lower r to save memory
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 5. SFT Configuration with Early Stopping settings
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=10, # Up to 10 epochs
    eval_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch", # Save at the end of each epoch
    load_best_model_at_end=True, # Required for early stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_steps=-1,
    fp16=False, # Disable AMP to prevent bf16 scaling issues on GTX 1650
    bf16=False,
    optim="paged_adamw_8bit",
    report_to="none",
    dataset_text_field="text",
    max_length=512,
    gradient_checkpointing=True,
)

# 6. SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stop if no improvement after 2 epochs
)

# 7. Start Training
print("Starting training with Early Stopping...")
trainer.train()

# 8. Save final best model
print(f"Saving best model to {final_model_dir}")
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("Training complete!")
