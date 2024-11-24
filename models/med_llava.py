datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'

cache_dir = "eepy/llava-med-v1.5-mistral-7b"

from transformers import LlamaForCausalLM, AutoTokenizer, MistralForCausalLM


# Query tokenizer and model from cache
tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)

#model = LlamaForCausalLM.from_pretrained(cache_dir, local_files_only=True, device_map="auto")
model = MistralForCausalLM.from_pretrained(cache_dir, local_files_only=True, device_map="auto")