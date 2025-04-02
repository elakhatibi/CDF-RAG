from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# base_model = "meta-llama/Llama-2-7b-hf"
# ✅ NEW (open-access):
try:
    base_model = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
except:
    print("⚠️ LLaMA-2 gated model not available. Using fallback...")
    base_model = "NousResearch/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files="/home/hao/colm/dataset for fine-tuning/cdf_query_refinement_600.jsonl", split="train")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def tokenize(example):
    prompt = format_prompt(example)
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # <-- required for computing loss
    return tokenized



tokenized = dataset.map(tokenize)

# Load and prepare model for LoRA
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

# PEFT config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Training
training_args = TrainingArguments(
    output_dir="./models/llama-cdf",
    per_device_train_batch_size=2,
    num_train_epochs=20,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()
