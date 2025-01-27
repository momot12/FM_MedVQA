import os
import torch
from transformers import TrainingArguments, Trainer, LlavaForConditionalGeneration, LlavaProcessor
from datasets import load_dataset
from peft import PeftModel, PeftConfig  # LoRA utilities
import datetime
from PIL import Image
from torchvision import transforms
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_cache_dir', type=str, default='data/data_cache', help='Directory to cache datasets')
parser.add_argument('--model_cache_dir', type=str, default='data/model_cache', help='Directory to cache models')
parser.add_argument('--sample', action='store_true', help='Use sample dataset')
parser.add_argument('--output_dir', type=str, default='output/llava-peft/ckpt', help='Output directory') # TODO: check path
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
args = parser.parse_args()


# Path to directory
data_cache_dir = "data/data_cache"
model_cache_dir = "data/model_cache"
pretrain_lora_ckpt = 'output/llava-peft/ckpt' #TODO: create path
pretrain_lora_name = '' #TODO: check
pretrain_lora_ckpt = os.path.join(pretrain_lora_ckpt, pretrain_lora_name)
output_dir = 'output/llava-peft-finetune/ckpt'
# Save the model
now = datetime.datetime.now()
date_and_time = now.strftime("%Y-%m-%d-%H:%M")
output_dir = os.path.join(output_dir, date_and_time)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VQA-RAD dataset
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=data_cache_dir)

# Paths for processor, pre-trained model, and LoRA checkpoint
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=args.model_cache_dir)
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=args.model_cache_dir)


# Load LoRA configuration and apply it to the model
# lora_config = PeftConfig.from_pretrained(pretrain_lora_ckpt)
model = PeftModel.from_pretrained(base_model, pretrain_lora_ckpt)
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

model.train() # idk why it does not work, have to do param.requires_grad = True if 'lora' in name
model.to(device)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def init_wandb(args):
    # Initialize WandB run
    wandb.init(
        project="PEFT-ViLT-finetune-vqa-rad",
        entity="2024FM-MedVQA",
        config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": 5e-5,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "user": "cwlin"
        }
    )

# Preprocess the dataset
def preprocess(example):
    image = image_transform(example['image'])
    max_seq_length = model.config.max_position_embeddings

    encoding = processor(
        images=image,
        text=example["question"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    # Assuming the answer is a single string. Convert it to a label index or tensor.
    encoding["labels"] = torch.tensor(processor.tokenizer.encode(example["answer"], padding="max_length", max_length=max_seq_length, truncation=True))
    return {key: value.squeeze(0) for key, value in encoding.items()}  # Remove extra batch dimension

# Apply preprocessing to the dataset
train_dataset = ds_vqa_rad["train"].map(preprocess, remove_columns=ds_vqa_rad["train"].column_names)
test_dataset = ds_vqa_rad["test"].map(preprocess, remove_columns=ds_vqa_rad["test"].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="wandb",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
)

# Start training
trainer.train()
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=2)

# for epoch in range(10):
#     for batch in train_loader:
#         # Inspect the first batch

#         print(f"Pixel Values batch shape: {batch['pixel_values'].shape}")
#         print(f"Input IDs batch shape: {batch['input_ids'].shape}")
#         print(f"Labels batch shape: {batch['labels'].shape}")
#         break

model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir) # should I save tokenizer?
wandb.finish()
