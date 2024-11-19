from datasets import load_dataset

cache_dir = 'eepy/datasets/'
# VQA-RAD
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)

# SLAKE English
ds_slake = load_dataset("mdwiratathya/SLAKE-vqa-english", cache_dir=cache_dir)

# PathVQA
ds_pathVQA = load_dataset("flaviagiammarino/path-vqa", cache_dir=cache_dir)