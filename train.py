import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import DPOTrainer

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clean up garbage
def clean():
    gc.collect()
    torch.cuda.empty_cache()

clean()


# Select your model
model_name = "Sharathhebbar24/SSH_355M"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Dataset for DPO
dataset_name = "Intel/orca_dpo_pairs"
dataset = load_dataset(dataset_name, split="train")
num_rows = dataset.num_rows

# Training Arguments
batch_size = 2
max_steps = 100
training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=batch_size,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    max_steps=max_steps,
    save_strategy="no",
    logging_steps=50,
    output_dir="./models/dpo/",
    warmup_steps=max_steps//4,
    fp16=True,
)

# Create DPO trainer
max_prompt_length = 512
max_length = 1024
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
    max_prompt_length=max_prompt_length,
    max_length=max_length,
    
)

# Fine-tune model with DPO
dpo_trainer.train()

new_model = "DPO_FineTuned_SSH_355M"
HF_TOKEN = "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"

tokenizer.push_to_hub(
    new_model,
    token=HF_TOKEN
)

model.push_to_hub(
    new_model,
    token=HF_TOKEN
)