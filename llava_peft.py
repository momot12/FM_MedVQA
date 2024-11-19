datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'

import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AdamW, AutoProcessor, LlavaForConditionalGeneration
from peft import get_peft_model, LoraConfig
import wandb


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ROCOv2Dataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, image_transform, negative_sample_ratio=0.5, max_length=40):
        """
        Args:
            image_dir (str): Directory path containing images.
            caption_file (str): Path to the CSV file containing captions.
            tokenizer: Hugging Face tokenizer.
            image_transform: Transformations applied to images.
            negative_sample_ratio: Ratio of negative samples (non-matching pairs).
        """
        self.image_dir = image_dir
        self.captions_df = pd.read_csv(caption_file)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.negative_sample_ratio = negative_sample_ratio
        self.max_length = max_length

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        # Get the image ID and caption
        row = self.captions_df.iloc[idx]
        image_id = row['ID']
        caption = row['Caption']
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate label for ITM: 1 for matching, 0 for non-matching
        label = 1
        if random.random() < self.negative_sample_ratio:
            label = 0
            # Sample a negative caption
            neg_idx = random.randint(0, len(self.captions_df) - 1)
            negative_caption = self.captions_df.iloc[neg_idx]['Caption']
            text_inputs = self.tokenizer(
                negative_caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

        # Flatten to remove the batch dimension
        text_inputs = {key: val.squeeze(0) for key, val in text_inputs.items()}

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'label': torch.tensor(label)
        }

# load model and processor
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_cache_path)
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_cache_path)

# Define paths
train_image_dir = 'sample_datasets/ROCOv2/train'
train_caption_file = 'sample_datasets/ROCOv2/sample_train_captions.csv'
valid_image_dir = 'sample_datasets/ROCOv2/valid'
valid_caption_file = 'sample_datasets/ROCOv2/sample_valid_captions.csv'
test_image_dir = 'sample_datasets/ROCOv2/test'
test_caption_file = 'sample_datasets/ROCOv2/sample_test_captions.csv'


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset instances
train_dataset = ROCOv2Dataset(train_image_dir, train_caption_file, processor, image_transform)
valid_dataset = ROCOv2Dataset(valid_image_dir, valid_caption_file, processor, image_transform)
test_dataset = ROCOv2Dataset(test_image_dir, test_caption_file, processor, image_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define LoRA configuration
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj"],  # LLaVA's attention layers
    task_type='vision_language',
    r=8,  # Low-rank dimension
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1  # Dropout for LoRA layers
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss function for ITM (binary classification)
loss_fn = nn.CrossEntropyLoss()  # or nn.BCEWithLogitsLoss() if your labels are floats

output_dir = 'eepy/llava-lora/output'
logging_dir = 'eepy/llava-lora/logging'

from tqdm import tqdm
import torch

# Set model to training mode
model.train()
model.to(device)

# Number of epochs
num_epochs = 10  # Adjust based on your dataset size and compute resources

# Initialize WandB run
wandb.init(
    project="PEFT-LLaVA-ITM", # image-text-matching
    entity="cwlin",
    config={
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": 5e-5,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1
    }
)


# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0
    for batch in tqdm(train_loader):
        # Move batch data to device
        images = batch['image'].to(device)
        captions = batch['caption']  # Captions remain as strings for processor
        labels = batch['label'].to(device)

        # Preprocess inputs
        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Forward pass
        outputs = model(**inputs)

        # ITM logits for binary classification
        logits = outputs.logits[:, 0, :2]  # Adjust this indexing based on output format
        
        loss = loss_fn(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Log loss to WandB
        wandb.log({"loss": loss.item()})

    avg_loss = epoch_loss / len(train_loader)
    print(f"Average training loss for epoch {epoch + 1}: {avg_loss:.4f}")

    # Log average loss for the epoch
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})


# Save model after training
#model.save_pretrained(output_dir)

# example
"""
inputs = processor(
    text=["A cat sitting on a mat"],
    images=torch.randn(1, 3, 224, 224),  # Example preprocessed image
    return_tensors="pt"
)
outputs = model(**inputs)
"""