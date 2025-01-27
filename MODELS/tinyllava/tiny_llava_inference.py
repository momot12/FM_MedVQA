"""Inference on vqa-rad, slake, pathVQA."""
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import torch


# Empty cache to avoid cuda issues
torch.cuda.empty_cache()
print('Emptied cuda cache.')

# Put on GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

MODEL_CACHE = '/mount/studenten-temp1/users/takamo/FM24vqa/model_cache'
OUTPUT = 'OUTPUTS_jsonl/'
DATASETS = ['VQA-RAD', 'SLAKE', 'PathVQA']
MAX_LENGTH_GEN = 1024
# change to 0-VQA-RAD, 1-SLAKE, 2-PathVQA
DS = DATASETS[2]

model = AutoModelForImageTextToText.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir=MODEL_CACHE).to(device)
processor = AutoProcessor.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir=MODEL_CACHE)

print('*** Loaded model and processor. ***\n\n*** Starting inference ***')

from tqdm import tqdm
import json
output_file = open(OUTPUT+f'{MAX_LENGTH_GEN}_tinyllava_test_{DS.lower()}_answer_pred.jsonl', 'w')
output_file.write('[\n')

lines = json.load(open(f'data/{DS}/test_question_answer_gt.jsonl', 'r'))
outputs = []
for idx, data in tqdm(enumerate(lines)):
    prompt = 'USER: <image>\n'
    question = data['question']
    prompt += f'{question} ASSISTANT:'
    image = Image.open(f'data/{DS}/test/{data["id"]}.png')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH_GEN)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    answer = output.split('ASSISTANT: ')[1]
    if idx == len(lines) - 1:
        output_file.write(json.dumps({
            'question_id': data['id'],
            'question': question,
            "question_type": data["question_type"],
            'answer_pred': answer,
        }) + ']')
    else:
        output_file.write(json.dumps({
            'question_id': data['id'],
            'question': data['question'],
            "question_type": data["question_type"],
            'answer_pred': answer
        }) + ',\n')

print('\n*** Inference done. ***')