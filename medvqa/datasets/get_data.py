from datasets import load_dataset

dataset_cache_dir = '/home/st/st_us-053000/st_st191423/2024WS-FM/datasets/'
# VQA-RAD
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=dataset_cache_dir)
print('loaded vqa rad')

# SLAKE English
ds_slake = load_dataset("mdwiratathya/SLAKE-vqa-english", cache_dir=dataset_cache_dir)
print('loaded slake')

# PathVQA
ds_pathVQA = load_dataset("flaviagiammarino/path-vqa", cache_dir=dataset_cache_dir)
print('loaded pathvqa')