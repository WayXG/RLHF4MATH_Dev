import os
from datasets import load_dataset

# 设置你的文件夹路径
all_folder_path = [#'/home/wexiong_google_com/wx/ToRA_math/src/output1/llm-agents/tora-code-34b-v1.0/math',
        #'/home/wexiong_google_com/wx/ToRA/src/output1/gemma-2b-it_bs128_lr1e-5/checkpoint-3511/math',
        #'/home/wexiong_google_com/wx/ToRA/src/output1/gemma-2b-it_bs128_lr1e-5/checkpoint-5265/math',
        #'/home/wexiong_google_com/wx/ToRA/src/iter2_gen_data/dpo_plus_1nll_iter1/checkpoint-300/math'
        #'/home/wexiong_google_com/wx/ToRA/src/sft_gen_data_gsm8k/1231czx/2b_sft2/gsm8k'
        #'/home/wexiong_google_com/wx/ToRA/src/iter2_gen_data_math/dpo_iter1/checkpoint-600/math'
        #'/home/wexiong_google_com/wx/ToRA/src/sft_gen_data_gsm8k/gemma-2b-it_bs128_lr1e-5/checkpoint-3511/gsm8k'
        #'/home/wexiong_google_com/wx/ToRA/src/iter2_gen_data_gsm8k/dpo_iter1/checkpoint-250/gsm8k'
        '/home/wexiong_google_com/wx/ToRA/src/7b_sft3epoch_gen_data/1231czx/7b_510k_5e6_bz64_sft3peoch/gsm8k'
        ]
        #'/home/wexiong_google_com/wx/ToRA/src/output1/llm-agents/tora-code-34b-v1.0/gsm8k']
output_dir='all_math.json'


#jsonl_files = [folder_path + '/' + f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
#print(jsonl_files)

all_data = []
for folder_path in all_folder_path:
    jsonl_files = [folder_path + '/' + f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    for dir_ in jsonl_files:
        ds_test = load_dataset('json', data_files=dir_, split='train')
        for sample in ds_test:
            all_data.append(sample)


import json

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_data
print("I collect ", len(all_data), "samples")


with open(output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

z = load_dataset("json", data_files=output_dir, split='train', field='instances')
z.push_to_hub('1231czx/7b_sft_510k_3epoch_gen_data_iter1')
~                                                                                                                                                
~                                                                      
