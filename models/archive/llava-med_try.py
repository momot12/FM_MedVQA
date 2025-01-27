import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, MistralForCausalLM

# Path to the model
cache_dir = "/mount/studenten-temp1/users/takamo/llava-med-v1.5-mistral-7b"
cache_ds = "/mount/studenten/team-lab-cl/data2024/fm_med_vqa/momo_fm/FM_MedVQA/data"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### MODEL and TOKENIZER ###
tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
model = MistralForCausalLM.from_pretrained(cache_dir, local_files_only=True, device_map="auto")

model.eval()  # Set model to evaluation mode

print(f"*** Finished querying tokenizer and llava-med model. ***\n\n CAUTION: eval mode\n\n*** Prepping data. ***")

### DATASET PROCESSING - VQA_RAD ###
vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_ds)

# Image transformations (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to expected dimensions
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to preprocess data for inference
def preprocess_data(image, question):
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Preprocess the question (tokenize)
    text_inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

    return image_tensor, text_inputs


print(f'Dataset: VQA-RAD')

### TESTING ###
test_data = vqa_rad["test"][0]
predictions = []

# Extract questions, answers, and images from the dataset
questions = [item["question"] for item in test_data]
answers = [item["answer"] for item in test_data]
images = [Image.open(item["image"]) for item in test_data]  # Assuming the image is stored as a path or PIL Image


predictions = []

print(f"\n*** Start TESTING ***")
predictions = []

for image, question, ground_truth in zip(images, questions, answers):
    # Preprocess the image and question
    image_tensor, text_inputs = preprocess_data(image, question)

    # Generate output with the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=text_inputs['input_ids'],  # Pass tokenized question
            attention_mask=text_inputs['attention_mask'],  # Pass attention mask
            pixel_values=image_tensor,  # Pass preprocessed image
            max_new_tokens=50,  # Max tokens to generate
            pad_token_id=tokenizer.eos_token_id  # Pad token id
        ) 
    
    # Decode the model's output (convert token IDs back to text)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(predicted_answer)

# Now `predictions` contains the model's answers
print(predictions)


