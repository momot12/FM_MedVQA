datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'

from transformers import TrainingArguments, Trainer
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType

# load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=hf_cache_path)
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=hf_cache_path)


peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# Define LoRA configuration
lora_config = LoraConfig(
    task_type='vision_language',
    r=8,  # Low-rank dimension
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1  # Dropout for LoRA layers
)

model = get_peft_model(model, lora_config)

output_dir = 'eepy/vilt-lora/output'
logging_dir = 'eepy/vilt-lora/logging'
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir=logging_dir,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2
)

class ROCOv2Dataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, transform=None, image_size=(384, 384)):
        self.image_dir = image_dir
        self.captions = pd.read_csv(caption_file)
        self.processor = processor
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Get image and caption data
        img_id = self.captions.iloc[idx, 0]
        caption = self.captions.iloc[idx, 1]
        
        # Load and resize image
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB").resize(self.image_size)

        # Process inputs with padding
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        
        # Flatten dictionary of tensors to return as batch-friendly format
        return {k: v.squeeze(0) for k, v in inputs.items()}



# Paths
train_image_dir = 'sample_datasets/ROCOv2/train'
valid_image_dir = 'sample_datasets/ROCOv2/valid'
test_image_dir = 'sample_datasets/ROCOv2/test'

train_caption_file = 'sample_datasets/ROCOv2/sample_train_captions.csv'
valid_caption_file = 'sample_datasets/ROCOv2/sample_valid_captions.csv'
test_caption_file = 'sample_datasets/ROCOv2/sample_test_captions.csv'

# Instantiate processor
from transformers import ViltProcessor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=hf_cache_path)

# Datasets and DataLoaders
train_dataset = ROCOv2Dataset(train_image_dir, train_caption_file, processor)
valid_dataset = ROCOv2Dataset(valid_image_dir, valid_caption_file, processor)
test_dataset = ROCOv2Dataset(test_image_dir, test_caption_file, processor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=4)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor
)

trainer.train()
trainer.evaluate(test_dataset)
model.save_pretrained("./vilt_lora_finetuned")