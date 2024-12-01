import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class ROCOv2Dataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, image_transform, max_length=40):
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
        self.processor = processor
        self.image_transform = image_transform
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
        inputs = self.processor(image, caption, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),  # Preprocessed image
            'input_ids': inputs['input_ids'].squeeze(0),        # Preprocessed text
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }