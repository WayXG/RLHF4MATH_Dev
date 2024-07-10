import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = '/home/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/llama3-8b-it_bs64_lr5e6/checkpoint-1254'
output_name = 'xxx/llama38b_5e6_sft3epoch'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model.push_to_hub(output_name)
tokenizer.push_to_hub(output_name)
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                                                             
