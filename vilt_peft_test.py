import torch
from transformers import ViltProcessor, ViltForMaskedLM, AutoTokenizer
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from peft import PeftModel

# Load paths and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directories
data_cache_dir = "data/data_cache"
model_cache_dir = "data/model_cache"
finetuned_model_path = 'output/vilt-peft-finetune/ckpt/2024-11-27-16:38'

# Load the VQA-RAD dataset (or another test dataset)
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=data_cache_dir)

# Paths for processor, pre-trained model, and LoRA checkpoint
vilt_mlm = "dandelin/vilt-b32-mlm"
processor = ViltProcessor.from_pretrained(vilt_mlm, cache_dir=model_cache_dir)
tokenizer = AutoTokenizer.from_pretrained(vilt_mlm, cache_dir=model_cache_dir)
base_model = ViltForMaskedLM.from_pretrained(vilt_mlm, cache_dir=model_cache_dir)

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the image preprocessing transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example test loop (using the first example from the dataset)
test_example = ds_vqa_rad['test'][0]  # Example from the test set
image = test_example['image'].convert("RGB")
# image = Image.open(test_example['image']).convert("RGB")
question = test_example['question']

# Preprocess image and text
inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    predicted_ids = outputs.logits.argmax(dim=-1)

# Decode the output
predicted_answer = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

# Output the predicted answer
print(f"Question: {question}")
print(f"Predicted Answer: {predicted_answer}")
