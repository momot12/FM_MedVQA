PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/images \
    --answers-file /path/to/answer-file.jsonl \
    --temperature 0.0