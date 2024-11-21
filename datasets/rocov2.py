import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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