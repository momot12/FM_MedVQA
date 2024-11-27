import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

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