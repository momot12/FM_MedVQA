import json
import pandas as pd
import os

sample_dict = dict()

for split in ['train', 'valid', 'test']:
    sample_dict[split] = []
    split_path = f'eepy/datasets/ROCOv2/{split}'
    print(len(os.listdir(split_path)))

    for f in os.listdir(split_path):
        sample_dict[split].append(f.split('.jpg')[0])

sample_dict['train'] = sample_dict['train'][:600]
sample_dict['valid'] = sample_dict['valid'][:100]
sample_dict['test'] = sample_dict['test'][:100]

data_dict = dict()
for split in ['train', 'valid', 'test']:
    data_dict[split] = dict()
    for type in ['captions', 'concepts', 'concepts_manual']:
        # with open(f'eepy/datasets/ROCOv2/{split}_{type}.json', 'r') as f:
        data = pd.read_csv(f'eepy/datasets/ROCOv2/{split}_{type}.csv')
        data_dict[split][type] = data

sample_data_dict = dict()
for split in ['train', 'valid', 'test']:
    sample_data_dict[split] = dict()
    for type in ['captions', 'concepts', 'concepts_manual']:
        sample_data_dict[split][type] = dict()
        for sample_id in sample_dict[split]:
            columns = data_dict[split][type][data_dict[split][type]['ID'] == sample_id].columns
            sample_data_dict[split][type][data_dict[split][type][data_dict[split][type]['ID'] == sample_id][columns[0]].values[0]] = data_dict[split][type][data_dict[split][type]['ID'] == sample_id][columns[1]].values[0]

for split in ['train', 'valid', 'test']:
    for type in ['captions', 'concepts', 'concepts_manual']:
        _df = pd.DataFrame.from_dict(sample_data_dict[split][type], orient='index')
        _df.reset_index(inplace=True)
        _df.columns = data_dict[split][type].columns
        _df.to_csv(f'sample_datasets/ROCOv2/sample_{split}_{type}.csv', index=False)

# copy images from eepy to sample_datasets
import shutil

import json
import pandas as pd
import os

sample_dict = dict()

for split in ['train', 'valid', 'test']:
    sample_dict[split] = []
    split_path = f'eepy/datasets/ROCOv2/{split}'
    print(len(os.listdir(split_path)))

    for f in os.listdir(split_path):
        sample_dict[split].append(os.path.join(split_path, f))

sample_dict['train'] = sample_dict['train'][:600]
sample_dict['valid'] = sample_dict['valid'][:100]
sample_dict['test'] = sample_dict['test'][:100]

for split in ['train', 'valid', 'test']:
    output_dir = f'sample_datasets/ROCOv2/{split}'
    os.makedirs(output_dir, exist_ok=True)

    for file_path in sample_dict[split]:
        shutil.copy(file_path, output_dir)