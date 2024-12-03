# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import torch

MODEL_CACHE = '/mount/studenten-temp1/users/takamo/FM24vqa/model_cache'


processor = AutoProcessor.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir =MODEL_CACHE)
model = AutoModelForImageTextToText.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir =MODEL_CACHE)


# Example: Load VQA-RAD data
vqa_rad_data_path = "path_to_vqa_rad_dataset.json"  # Update with the correct path
with open(vqa_rad_data_path, "r") as file:
    vqa_data = json.load(file)

# Example of preprocessing and inference
for entry in vqa_data:
    image_path = entry['image_path']  # Path to the image
    question = entry['question']     # Associated question

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt")

    # Generate answer
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    print(f"Question: {question}")
    print(f"Answer: {answer}")