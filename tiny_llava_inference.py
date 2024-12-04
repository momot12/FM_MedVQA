# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import torch


torch.cuda.empty_cache()
print('Emptied cuda cache.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

MODEL_CACHE = '/mount/studenten-temp1/users/takamo/FM24vqa/model_cache'
OUTPUT = '/mount/studenten-temp1/users/takamo/FM24vqa/outputs/'

model = AutoModelForImageTextToText.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir=MODEL_CACHE).to(device)
processor = AutoProcessor.from_pretrained("bczhou/tiny-llava-v1-hf", cache_dir=MODEL_CACHE)

print('*** Loaded model and processor. ***\n\n*** Starting inference ***')

from tqdm import tqdm
import json
output_file = open(OUTPUT+'llava_outputs/tinyllava_test_vqa_rad_answer_pred.jsonl', 'w')
output_file.write('[\n')

lines = json.load(open('data/VQA-RAD/test_question_answer_gt.jsonl', 'r'))
outputs = []
for idx, data in tqdm(enumerate(lines)):
    prompt = 'USER: <image>\n'
    question = data['question']
    prompt += f'{question} ASSISTANT:'
    image = Image.open(f'data/VQA-RAD/test/{data["id"]}.png')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=15)
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