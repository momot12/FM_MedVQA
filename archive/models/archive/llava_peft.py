import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from transformers import AdamW, LlavaProcessor, LlavaForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from peft import get_peft_model, LoraConfig
import wandb
import sys
print(os.getcwd())
sys.path.append('/home/users1/linchi/2024WS-FM/FM_MedVQA')
from medvqa.datasets.llava.datasets import ROCOv2Dataset
import argparse
from tqdm import tqdm
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


parser = argparse.ArgumentParser()
parser.add_argument('--data_cache_dir', type=str, default='data/data_cache', help='Directory to cache datasets')
parser.add_argument('--model_cache_dir', type=str, default='data/model_cache', help='Directory to cache models')
parser.add_argument('--sample', action='store_true', help='Use sample dataset')
parser.add_argument('--image_dir', type=str, default='data/ROCOv2', help='Directory containing images and captions')
parser.add_argument('--output_dir', type=str, default='output/llava-peft/ckpt', help='Output directory')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--device_id', type=int, help='GPU device ID')
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
    ])
    # Create dataset instances
    train_dataset = ROCOv2Dataset(train_image_dir, train_caption_file, tokenizer, image_transform)
    # valid_dataset = ROCOv2Dataset(valid_image_dir, valid_caption_file, tokenizer, image_transform)
    # test_dataset = ROCOv2Dataset(test_image_dir, test_caption_file, tokenizer, image_transform)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print(f"Train dataset: {len(train_dataset)} samples")
    # print(f"Valid dataset: {len(valid_dataset)} samples")
    # print(f"Test dataset: {len(test_dataset)} samples")
    return train_loader # , valid_loader, test_loader

def init_wandb(num_epochs, train_loader):
    # Initialize WandB run
    wandb.init(
        project="PEFT-Llava-pretrain-conditional-generation",
        entity="cwlin",     # Replace with your WandB username or team
        config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": 5e-5,
            "lora_r": 2,
            "lora_alpha": 4,
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
        target_modules=["q_proj", "k_proj", "v_proj"],  # LLaVA's attention layers
        task_type='vision_language',
        r=2,  # Low-rank dimension
        lora_alpha=4,  # Scaling factor
        lora_dropout=0.1  # Dropout for LoRA layers
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(args, num_epochs, train_loader, model, optimizer, device):
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        for batch in tqdm(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            del pixel_values, input_ids, attention_mask
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        # Log loss to WandB
        wandb.log({"train_loss": epoch_loss})
    
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
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        device = torch.device(f"cuda:{args.device_id}")
    print(f"Using device {device}")

    # load model and processor
    llava_hf = "llava-hf/llava-1.5-7b-hf"
    tiny_llava_hf = "bczhou/tiny-llava-v1-hf"
    # processor = LlavaProcessor.from_pretrained(llava_hf, cache_dir=args.model_cache_dir)
    # model = LlavaForConditionalGeneration.from_pretrained(llava_hf, cache_dir=args.model_cache_dir, device_map='auto')

    processor = AutoProcessor.from_pretrained(tiny_llava_hf, cache_dir=args.model_cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(tiny_llava_hf, cache_dir=args.model_cache_dir, device_map='auto')
    
    train_loader = load_dataset(args, processor)

    # Initialize WandB
    num_epochs = args.num_epochs
    # init_wandb(num_epochs, train_loader)

    # Apply LoRA to the model
    model = apply_lora_to_model(model)
    # model = torch.nn.DataParallel(model)
        
    # Set model to training mode
    # model.train()
    # model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train(args, num_epochs, train_loader, model, optimizer, device=device)


if __name__ == '__main__':
    main(args)