from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import json

data_cache_dir = "data/data_cache"
model_cache_dir = "./data/model_cache"
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=model_cache_dir)
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=model_cache_dir)

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

for dataset in ["VQA-RAD", "PathVQA", "SLAKE"]:
    lines = json.load(open(f'data/{dataset}/test_question_answer_gt.jsonl', 'r'))
    output_file = open(f'output/vilt_test_{dataset}_answer-file.jsonl', 'w')
    output_file.write('[\n')

    from tqdm import tqdm
    for idx, data in tqdm(enumerate(lines)):
        image = Image.open(f'data/{dataset}/test/{data["id"]}.png')
        if image.mode == 'L':  # 'L' mode is grayscale
            image = image.convert('RGB')
        question = data["question"]
        # encoding = processor(image, question, return_tensors="pt", padding="max_length", max_length=50, truncation=True)
        encoding = processor(image, question, return_tensors="pt", padding="max_length", max_length=40, truncation=True)
        outputs = model(**encoding)
        idx = outputs.logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        question_type, answer_type = classify_question_answer(question, answer)
        # print("Predicted answer:", model.config.id2label[idx],
            # "True answer:", data["answer"])

        if idx != len(lines) - 1:
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