"""Inference on finetuned TinyLLaVA with vqa-rad, slake, pathVQA.
   The script is taken from https://github.com/TinyLLaVA/TinyLLaVA_Factory/tree/tinyllava_bench"""
import argparse
import torch

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# Model
dataset_name = 'SLAKE'
model_path = f"/mount/studenten-temp1/users/linchi/2024WS-FM/output/tiny-llava/checkpoints/tiny-llava-base-phi-2-siglip-so400m-patch14-384-{dataset_name}-finetune"
model_base = None
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

def eval_model(args, model_name, tokenizer, model, image_processor, context_len):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

from tqdm import tqdm
import json
output_dir = '/mount/studenten-temp1/users/linchi/2024WS-FM/output/tiny-llava/answers'
dataset_dir = f"/mount/studenten-temp1/users/linchi/2024WS-FM/datasets/{dataset_name}"

with open(f"{dataset_dir}/test_question_answer_gt.jsonl", "r") as f:
    test_data = json.load(f)

all_args = []
for i in range(len(test_data)):
    prompt = test_data[i]['question']
    id = test_data[i]['id']
    image_file = f"{dataset_dir}/test/{id}.png"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": "phi",
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    all_args.append(args)

answers = []
for args in tqdm(all_args):
    answer = eval_model(args, model_name, tokenizer, model, image_processor, context_len)
    answers.append(answer)

def write_json(test_data, answers):

    # write a jsonl file
    with open(f"{output_dir}/test_{dataset_name}_tinyllava-finetune_pred.jsonl", "w") as f:
        f.write("[")
        for i in range(len(test_data)):
            json.dump({'question_id': test_data[i]['id'], 
                       'question': test_data[i]['question'],
                       'answer_gt': test_data[i]['answer'],
                       'answer_pred': answers[i],
                       'question_type': test_data[i]['question_type'],
                       'answer_type': test_data[i]['answer_type']
                      }, f)
            if i != len(test_data) - 1:
                f.write("\n,")
            else:
                f.write("]")

write_json(test_data, answers)