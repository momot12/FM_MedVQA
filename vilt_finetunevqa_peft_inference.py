import json
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from medvqa.datasets.vilt.custom_vilt import ViltForVQA  # Replace 'your_script' with the name of the Python file containing the ViltForVQA class definition
from tqdm import tqdm

# Paths
DATASET_NAME = "VQA-RAD"
TRAIN_JSON = f"data/{DATASET_NAME}/train_question_answer_gt.jsonl"
TEST_JSON = f"data/{DATASET_NAME}/test_question_answer_gt.jsonl"
MODEL_DIR = 'output/vilt-peft-finetune/ckpt/2025-01-08-02:38 epoch=10'
MODEL_PATH = f'{MODEL_DIR}/vilt_vqa_full_model.pth'
TOKENIZER_PATH = "dandelin/vilt-b32-finetuned-vqa"  # Pretrained tokenizer path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load(MODEL_PATH)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Define image transforms (same as training)
image_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Data
def load_json(file_path, split='train'):
    if split == 'train':
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f] if file_path.endswith(".jsonl") else json.load(f)
    else:
        with open(file_path, "r") as f:
            return json.load(f)

train_data_raw = load_json(TRAIN_JSON, split='train')
test_data_raw = load_json(TEST_JSON, split='test')

# Function to classify question and answer types
def classify_question_answer(question, answer):
    question_type = question.split()[0].upper()
    if question_type not in ['WHAT', 'IS', 'DOES', 'WHERE', 'ARE', 'HOW', 'DO']:
        question_type = 'OTHER'
    
    if 'yes' in answer.lower() or 'no' in answer.lower():
        answer_type = "CLOSED"
    else:
        answer_type = "OPEN"
    return question_type, answer_type

output_file = open(f'output/vilt_finetunevqa_test_{DATASET_NAME}_answer-file.jsonl', 'w')
output_file.write('[\n')
for idx, data in tqdm(enumerate(test_data_raw)):
    # Example Input
    split = 'test'
    image_path = f"data/{DATASET_NAME}/{split}/{data['id']}.png"
    question = data['question']

    print(f"Image Path: {image_path}")
    print(f"Question: {question}")
    
    # Preprocess the image
    image = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0)  # Add batch dimension

    # Tokenize the question
    inputs = tokenizer(question, padding="max_length", truncation=True, max_length=40, return_tensors="pt")

    # Prepare input for the model
    input_data = {
        "pixel_values": image.to(device),
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    # Make a prediction
    with torch.no_grad():
        outputs = model(**input_data)

    # Extract logits and apply softmax
    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=-1)

    # Map Answers to Labels
    all_answers = {example['answer'] for data in [train_data_raw, test_data_raw] for example in data}
    answer_map = {a: i for i, a in enumerate(all_answers)}

    # Decode the answer
    predicted_label = torch.argmax(probs, dim=-1).item()
    answer = {v: k for k, v in answer_map.items()}[predicted_label]

    print(f"Predicted answer: {answer} \n Ground Truth answer: {data['answer']}", )
    question_type, answer_type = classify_question_answer(question, answer)

    if idx != len(test_data_raw) - 1:
        output_file.write(json.dumps({
                'question_id': data['id'],
                'question': data['question'],
                "question_type": question_type,
                'answer_pred': answer,
                'asnwer_type': answer_type
            }) + ',\n')
    else:
        output_file.write(json.dumps({
            'question_id': data['id'],
            'question': data['question'],
            "question_type": question_type,
            'answer_pred': answer,
            'asnwer_type': answer_type
        }) + ']')

output_file.close()
