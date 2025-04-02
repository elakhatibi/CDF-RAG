from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Load dataset
dataset = load_dataset("json", data_files="cdf_multitask_dataset_1000.jsonl", split="train")
base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prompt template for instruction tuning
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def tokenize(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Required for loss
    return tokenized

tokenized = dataset.map(tokenize)

# Load Mistral with quantization
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

# Add LoRA adapter
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Training
args = TrainingArguments(
    output_dir="./models/mistral-cdf",
    per_device_train_batch_size=2,
    num_train_epochs=20,
    learning_rate=3e-4,
    fp16=True,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()
