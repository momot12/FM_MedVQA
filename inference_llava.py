from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_cache_dir = './data/model_cache'
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=model_cache_dir)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=model_cache_dir)

from tqdm import tqdm
import json
output_file = open('output/test_vqa_rad_answer-file.jsonl', 'w')
output_file.write('[\n')

lines = json.load(open('data/VQA-RAD/test_question_answer_gt.jsonl', 'r'))
outputs = []
for idx, data in tqdm(enumerate(lines)):
    print(idx)
    prompt = 'USER: <image>\n'
    question = data['question']
    prompt += f'{question} ASSISTANT:'
    image = Image.open(f'data/VQA-RAD/test/{data["id"]}.png')
    inputs = processor(images=image, text=prompt, return_tensors="pt")
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