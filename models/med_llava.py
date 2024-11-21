datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/llava-med-v1.5-mistral-7b", cache_dir=hf_cache_path)