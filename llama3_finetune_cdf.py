from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import os

# Avoid tokenizer deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try loading LLaMA 3-8B (ensure your HuggingFace token grants access)
try:
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
except Exception as e:
    print("⚠️ LLaMA 3-8B not available or access denied. Please ensure you have access via HuggingFace.")
    raise e

tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files="/home/hao/colm/dataset for fine-tuning/cdf_query_refinement_600.jsonl", split="train")

# Prompt formatting
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

# Tokenization function
def tokenize(example):
    prompt = format_prompt(example)
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize dataset
tokenized = dataset.map(tokenize)

# Load base model in 4-bit mode and prepare for QLoRA
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

# LoRA config for PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, peft_config)

# Training configuration
training_args = TrainingArguments(
    output_dir="./models/llama3-cdf",
    per_device_train_batch_size=2,
    num_train_epochs=20,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_dir="./logs"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train the model
trainer.train()
