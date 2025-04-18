from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load & tokenize
dataset = load_dataset("json", data_files="cdf_multitask_dataset_1000.jsonl", split="train")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def preprocess(example):
    full_input = f"{example['instruction'].strip()}\n\n{example['input'].strip()}"
    return tokenizer(full_input, text_target=example["output"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(preprocess)

# Load model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Training args
args = TrainingArguments(
    output_dir="./models/flan-cdf",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=5e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train
trainer.train()
