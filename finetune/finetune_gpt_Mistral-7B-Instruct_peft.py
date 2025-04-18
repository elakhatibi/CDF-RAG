from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
dataset = load_dataset("json", data_files="/home/hao/colm/dataset for fine-tuning/cdf_query_refinement_600.jsonl", split="train")

# Model + tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Important for causal models

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Format dataset for causal LM
def preprocess(example):
    prompt = f"<s>[INST] {example['instruction'].strip()}\n\n{example['input'].strip()} [/INST] {example['output'].strip()}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# PEFT/LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/mistral-cdf",
    num_train_epochs=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    bf16=True,  # Use bf16 or fp16 if supported
    report_to="none"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # tokenizer=tokenizer
)


# Train
trainer.train()
