# Imports
import os
import json
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    ViltForMaskedLM,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel
import wandb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="SLAKE")
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()


# Constants and Configurations
DATA_CACHE_DIR = "data/data_cache"
MODEL_CACHE_DIR = "data/model_cache"
DATASET_NAME = args.dataset_name
TRAIN_JSON = f"data/{DATASET_NAME}/train_question_answer_gt.jsonl"
TEST_JSON = f"data/{DATASET_NAME}/test_question_answer_gt.jsonl"
OUTPUT_DIR_BASE = "output/vilt-peft-finetune/ckpt"
PRETRAIN_LORA_CKPT = "output/vilt-peft/ckpt/2024-11-27-06:29"
VILT_MLM = "dandelin/vilt-b32-mlm"
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epoch

# Create timestamped output directory
now = datetime.datetime.now()
date_and_time = now.strftime("%Y-%m-%d-%H:%M")
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, date_and_time, f"bs_{BATCH_SIZE}_ep_{EPOCHS}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Definition
class ViltForVQA(nn.Module):
    def __init__(self, peft_model, num_labels):
        super().__init__()
        self.vilt = peft_model.base_model.model.vilt
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.vilt.config.hidden_size, self.vilt.config.hidden_size * 2),
            nn.LayerNorm(self.vilt.config.hidden_size * 2, eps=1e-5),
            nn.GELU(),
            nn.Linear(self.vilt.config.hidden_size * 2, num_labels)
        )

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.vilt(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

# Load Data
# def load_json(file_path, split='train'):
#     if split == 'train':
#         with open(file_path, "r") as f:
#             return [json.loads(line) for line in f] if file_path.endswith(".jsonl") else json.load(f)
#     else:
#         with open(file_path, "r") as f:
#             return json.load(f)
def load_json(file_path, split=None):
    with open(file_path, "r") as f:
        return json.load(f)

train_data_raw = load_json(TRAIN_JSON, split='train')
test_data_raw = load_json(TEST_JSON, split='test')

# Map Answers to Labels
all_answers = {example['answer'] for data in [train_data_raw, test_data_raw] for example in data}
answer_map = {a: i for i, a in enumerate(all_answers)}

# Tokenizer and Image Transform
tokenizer = AutoTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=MODEL_CACHE_DIR)
image_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset Preprocessing
def preprocess_dataset(data_raw, split):
    processed_data = []
    qid = "question_id" if split == "train" else "id"
    q = "text" if split == "train" else "question"
    for example in tqdm(data_raw):
        image_path = f"data/{DATASET_NAME}/{split}/{example[qid]}.png"
        image = image_transform(Image.open(image_path).convert("RGB"))
        inputs = tokenizer(
            example[q], padding="max_length", truncation=True, max_length=40, return_tensors="pt"
        )
        label = torch.tensor(answer_map.get(example["answer"], -1), dtype=torch.long)
        processed_data.append({
            "pixel_values": image,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label,
        })
    return processed_data

train_data = preprocess_dataset(train_data_raw, "train")
test_data = preprocess_dataset(test_data_raw, "test")

# Data Collator
def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    padded_input_ids = torch.stack([
        nn.functional.pad(item["input_ids"], (0, max_len - item["input_ids"].size(0))) for item in batch
    ])
    padded_attention_mask = torch.stack([
        nn.functional.pad(item["attention_mask"], (0, max_len - item["attention_mask"].size(0))) for item in batch
    ])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "input_ids": padded_input_ids, "attention_mask": padded_attention_mask, "labels": labels}

# DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Load Pretrained Model
base_model = ViltForMaskedLM.from_pretrained(VILT_MLM, cache_dir=MODEL_CACHE_DIR)
peft_model = PeftModel.from_pretrained(base_model, PRETRAIN_LORA_CKPT)
model = ViltForVQA(peft_model, num_labels=len(all_answers))

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
    report_to="wandb",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

# Main Script Execution
if __name__ == "__main__":
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    torch.save(model, f"{OUTPUT_DIR}/vilt_vqa_full_model.pth")
    wandb.finish()
