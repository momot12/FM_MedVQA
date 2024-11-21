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
from transformers import AutoTokenizer, ViltProcessor, ViltForMaskedLM, AdamW
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

    @staticmethod
    def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
        """
        Mask tokens for MLM.
        Args:
            input_ids: Tokenized input tensor with shape [sequence_length].
            tokenizer: Hugging Face tokenizer with mask token support.
            mlm_probability: Probability of masking a token.
        Returns:
            masked_input_ids: Input IDs with masked tokens.
            labels: Labels for MLM with -100 for non-masked tokens.
        """
        labels = input_ids.clone()  # Clone the tensor to create labels

        # Randomly mask tokens with a probability
        probability_matrix = torch.full(labels.shape, mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # We only mask non-special tokens
        special_tokens_mask = torch.tensor(
            tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            ),
            dtype=torch.bool
        )
        masked_indices &= ~special_tokens_mask

        # Set labels to -100 for unmasked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # Replace 80% of the masked tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.mask_token_id

        # Replace 10% of the masked tokens with random tokens
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        return input_ids, labels

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

        # Mask tokens for MLM
        masked_input_ids, mlm_labels = self.mask_tokens(
            text_inputs['input_ids'],  # Ensure tensor is passed
            self.tokenizer
        )

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'mlm_input_ids': masked_input_ids,
            'mlm_labels': mlm_labels,
            'label': torch.tensor(label)  # ITM label
        }



# load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", cache_dir=hf_cache_path)
model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm", cache_dir=hf_cache_path)

# Define paths
train_image_dir = 'sample_datasets/ROCOv2/train'
train_caption_file = 'sample_datasets/ROCOv2/sample_train_captions.csv'
valid_image_dir = 'sample_datasets/ROCOv2/valid'
valid_caption_file = 'sample_datasets/ROCOv2/sample_valid_captions.csv'
test_image_dir = 'sample_datasets/ROCOv2/test'
test_caption_file = 'sample_datasets/ROCOv2/sample_test_captions.csv'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dandelin/vilt-b32-mlm", cache_dir=hf_cache_path)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset instances
train_dataset = ROCOv2Dataset(train_image_dir, train_caption_file, tokenizer, image_transform)
valid_dataset = ROCOv2Dataset(valid_image_dir, valid_caption_file, tokenizer, image_transform)
test_dataset = ROCOv2Dataset(test_image_dir, test_caption_file, tokenizer, image_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define LoRA configuration
lora_config = LoraConfig(
    target_modules=['query', 'key', 'value'],
    task_type='vision_language',
    r=8,  # Low-rank dimension
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1  # Dropout for LoRA layers
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss functions
itm_loss_fn = nn.CrossEntropyLoss()  # ITM binary classification
mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore non-masked tokens

output_dir = 'eepy/vilt-lora/ckpt'
logging_dir = 'eepy/vilt-lora/logging'

from tqdm import tqdm
import torch

# Set model to training mode
model.train()
model.to(device)

# Number of epochs
num_epochs = 100  # Adjust based on your dataset size and compute resources

# Initialize WandB run
wandb.init(
    project="PEFT-ViLT-pretrain-mlm",     # Replace with your project name
    entity="cwlin",     # Replace with your WandB username or team
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
    epoch_itm_loss, epoch_mlm_loss = 0, 0

    for batch in tqdm(train_loader):
        # Move batch data to device (e.g., GPU)
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mlm_input_ids = batch['mlm_input_ids'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(
            pixel_values=images,
            input_ids=mlm_input_ids,  # Use masked input for MLM
            attention_mask=attention_mask
        )

        # ITM loss (from [CLS] token logits)
        itm_logits = outputs.logits[:, 0, :2]
        itm_loss = itm_loss_fn(itm_logits, labels)
        
        # MLM loss (from all logits)
        mlm_logits = outputs.logits
        mlm_loss = mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)),  # Flatten logits
            mlm_labels.view(-1)  # Flatten labels
        )

        # Combine losses
        loss = itm_loss + mlm_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_itm_loss += itm_loss.item()
        epoch_mlm_loss += mlm_loss.item()

        # Log loss to WandB
        wandb.log({"loss": loss.item()})

    avg_itm_loss = epoch_itm_loss / len(train_loader)
    avg_mlm_loss = epoch_mlm_loss / len(train_loader)

    print(f"Epoch {epoch + 1} - ITM Loss: {avg_itm_loss:.4f}, MLM Loss: {avg_mlm_loss:.4f}")

    # log average loss for the epoch
    wandb.log({"epoch": epoch + 1, "itm_loss": avg_itm_loss, "mlm_loss": avg_mlm_loss})

# Save the model
model.save_pretrained(output_dir)