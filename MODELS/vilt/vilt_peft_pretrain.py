"""This script is used for (1) medical domain adaptation (pre-training) for ViLT, by training only a LoRA adapter. 
   The pre-train objectives are image text match loss and mask language modeling loss."""
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from transformers import AutoTokenizer, ViltProcessor, ViltForMaskedLM, AdamW
from peft import get_peft_model, LoraConfig
import wandb
from tqdm import tqdm
import torch
from medvqa.datasets.vilt.datasets import ROCOv2Dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_cache_dir', type=str, default='data/data_cache', help='Directory to cache datasets')
parser.add_argument('--model_cache_dir', type=str, default='data/model_cache', help='Directory to cache models')
parser.add_argument('--sample', action='store_true', help='Use sample dataset')
parser.add_argument('--image_dir', type=str, default='data/ROCOv2', help='Directory containing images and captions')
parser.add_argument('--output_dir', type=str, default='output/vilt-peft/ckpt', help='Output directory')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
args = parser.parse_args()

def load_dataset(args, tokenizer):
    
    
    if args.sample:
        # Define paths
        print('Using sample dataset')
        train_image_dir = 'sample_datasets/ROCOv2/train'
        train_caption_file = 'sample_datasets/ROCOv2/sample_train_captions.csv'
        valid_image_dir = 'sample_datasets/ROCOv2/valid'
        valid_caption_file = 'sample_datasets/ROCOv2/sample_valid_captions.csv'
        test_image_dir = 'sample_datasets/ROCOv2/test'
        test_caption_file = 'sample_datasets/ROCOv2/sample_test_captions.csv'
    else:
        print('Using full dataset')
        # args
        train_image_dir = os.path.join(args.image_dir, 'train')
        train_caption_file = os.path.join(args.image_dir, 'train_captions.csv')
        valid_image_dir = os.path.join(args.image_dir, 'valid')
        valid_caption_file = os.path.join(args.image_dir, 'valid_captions.csv')
        test_image_dir = os.path.join(args.image_dir, 'test')
        test_caption_file = os.path.join(args.image_dir, 'test_captions.csv')


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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Valid dataset: {len(valid_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_loader, valid_loader, test_loader

def init_wandb(num_epochs, train_loader):
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

def apply_lora_to_model(model):
    """
    Apply LoRA to the model.
    Args:
        model: Base model to apply LoRA.
        lora_config: LoRA configuration.
    Returns:
        model: Model with LoRA applied.
    """
    # Define LoRA configuration
    lora_config = LoraConfig(
        target_modules=['query', 'key', 'value'],
        task_type='vision_language',
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1  # Dropout for LoRA layers
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(args, num_epochs, train_loader, model, optimizer, itm_loss_fn, mlm_loss_fn):
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
    # save datetime to the output directory
    import datetime
    now = datetime.datetime.now()
    date_and_time = now.strftime("%Y-%m-%d-%H:%M")
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, date_and_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def main(args):
    
    vilt_mlm = "dandelin/vilt-b32-mlm"
    # load model and processor
    processor = ViltProcessor.from_pretrained(vilt_mlm, cache_dir=args.model_cache_dir)
    model = ViltForMaskedLM.from_pretrained(vilt_mlm, cache_dir=args.model_cache_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(vilt_mlm, cache_dir=args.model_cache_dir)

    # Load the dataset
    train_loader, valid_loader, test_loader = load_dataset(args, tokenizer)
    
    # Initialize WandB
    num_epochs = args.num_epochs
    init_wandb(num_epochs, train_loader)
    
    # Apply LoRA to the model
    model = apply_lora_to_model(model)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Loss functions
    itm_loss_fn = nn.CrossEntropyLoss()  # ITM binary classification
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore non-masked tokens

    # Set model to training mode
    model.train()
    model.to(device)

    # Train the model
    train(args, num_epochs, train_loader, model, optimizer, itm_loss_fn, mlm_loss_fn)
if __name__ == '__main__':
    main(args)