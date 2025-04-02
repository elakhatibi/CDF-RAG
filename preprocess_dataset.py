from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(jsonl_path, model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def preprocess(example):
        input_text = f"{example['instruction'].strip()}\n\n{example['input'].strip()}"
        return tokenizer(input_text, text_target=example["output"], truncation=True, padding="max_length")

    return dataset.map(preprocess)
