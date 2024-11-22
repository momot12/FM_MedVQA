from datasets import load_dataset

cache_dir = 'eepy/datasets/'
# VQA-RAD
# Format --> {'image': PIL.JPEG, 'question': 'are regions of the brain infarcted?', 'answer': 'yes'}
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)

# SLAKE English
ds_slake = load_dataset("mdwiratathya/SLAKE-vqa-english", cache_dir=cache_dir)

print(ds_slake)

# PathVQA
ds_pathVQA = load_dataset("flaviagiammarino/path-vqa", cache_dir=cache_dir)


### PREPROCESSING DATA ### TODO