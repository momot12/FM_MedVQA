datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'


from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from transformers import AdamW, AutoProcessor, LlavaForConditionalGeneration
from peft import get_peft_model, LoraConfig
import wandb
from datasets.rocov2 import ROCOv2Dataset


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
ckpt_dir = 'eepy/llava-lora/ckpt'

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
model.save_pretrained(ckpt_dir)

# example
"""
inputs = processor(
    text=["A cat sitting on a mat"],
    images=torch.randn(1, 3, 224, 224),  # Example preprocessed image
    return_tensors="pt"
)
outputs = model(**inputs)
"""