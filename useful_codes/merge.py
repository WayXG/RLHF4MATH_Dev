# This script finds all the files ending with jsonl and merge them into one dataset 
import os
from datasets import load_dataset
import json

# The folders to load data
all_folder_path = [
        './7b_sft3epoch_gen_data/iter1',
        './7b_sft1epoch_gen_data/iter1'
]

output_dir='all_math.json'

all_data = []
for folder_path in all_folder_path:
    jsonl_files = [folder_path + '/' + f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    for dir_ in jsonl_files:
        ds_test = load_dataset('json', data_files=dir_, split='train')
        for sample in ds_test:
            all_data.append(sample)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_data
print("I collect ", len(all_data), "samples")


with open(output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
     
