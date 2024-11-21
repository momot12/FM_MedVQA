import torch
from transformers import ViltProcessor, ViltForMaskedLM, ViltForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel, PeftConfig  # LoRA utilities

# Path to directory
data_cache_dir = "eepy/datasets"
model_cache_dir = "eepy/hf-cache"
pretrain_lora_ckpt = 'eepy/vilt-lora/ckpt'
output_dir = 'eepy/vilt-lora/vqa_model_lora'
logging_dir = 'eepy/vilt-lora/logging'

# Load the VQA-RAD dataset
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=data_cache_dir)

# Paths for processor, pre-trained model, and LoRA checkpoint
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", cache_dir=model_cache_dir)
base_model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm", cache_dir=model_cache_dir)

# Load LoRA configuration and apply it to the model
lora_config = PeftConfig.from_pretrained(pretrain_lora_ckpt)
model = PeftModel.from_pretrained(base_model, pretrain_lora_ckpt)

# Preprocess the dataset
def preprocess(example):
    encoding = processor(
        images=example["image"],
        text=example["question"],
        padding="max_length",
        truncation=True,
        max_length=480,
        return_tensors="pt"
    )
    # Assuming the answer is a single string. Convert it to a label index or tensor.
    encoding["labels"] = torch.tensor(processor.tokenizer.encode(example["answer"], add_special_tokens=False))[:1]  # Take only first token
    return encoding

# Apply preprocessing to the dataset
train_dataset = ds_vqa_rad["train"].map(preprocess, remove_columns=ds_vqa_rad["train"].column_names)
test_dataset = ds_vqa_rad["test"].map(preprocess, remove_columns=ds_vqa_rad["test"].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
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
