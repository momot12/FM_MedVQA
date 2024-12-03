from datasets import load_dataset

dataset_cache_dir = '/mount/studenten-temp1/users/takamo/FM24vqa/data_cache'
# VQA-RAD
#ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=dataset_cache_dir)
#print('loaded vqa rad')

# SLAKE English
ds_slake = load_dataset("mdwiratathya/SLAKE-vqa-english", cache_dir=dataset_cache_dir)
#print('loaded slake')

print(len(ds_slake['test']['question']))
print(len(ds_slake['train']['question']))

# PathVQA
#ds_pathVQA = load_dataset("flaviagiammarino/path-vqa", cache_dir=dataset_cache_dir)
#print('loaded pathvqa')
