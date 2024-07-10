import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'mistralai/Mistral-7B-v0.3'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer))


model.save_pretrained("/home/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/mistral_with_pad")
tokenizer.save_pretrained("/home/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/mistral_with_pad")

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
~                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                       
~                      
