## LLaVA-Med scripts
The code refers to the original [LLaVA-Med Repo](https://github.com/microsoft/LLaVA-Med).   
The model checkpoint is downloaded via [huggingface hub](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b).  
Inference is done using a modified version of the [Multimodal Chat Inference](https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file#4-multimodal-chat-inference) code. See below:

```
export PYTHONPATH=$(pwd):$PYTHONPATH

DATASET=$1 # VQA-RAD, SLAKE, PathVQA
python3 llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file data/${DATASET}/llava_med-test_question_answer_gt.json \
    --image-folder data/${DATASET}/test \
    --answers-file data/${DATASET}/test-answer-file.jsonl \
    --temperature 0.0 
```