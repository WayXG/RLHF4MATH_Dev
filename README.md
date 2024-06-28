# RLHF4MATH_Dev

TL;DL: this is a repo to align the large language models (LLMs) by online iterative RLHF, with a focus on the multi-turn scenario and RL-free approaches. 

In particular, we consider the math problem solving with python interpreter, which means that the model can write a python code and ask the external environmnet to run and receive the excutaion result, before the LLM makes its next decision.

## Installation instructions

The main pipeline is divided into three steps:

- Supervised Fine-tuning (SFT), which is also a part of the RAFT algorithm;
- Data generation and annotation;
- DPO Training.

It is recommended to have three separate environments for **sft**, **inference**, and **dpo_train**. Please refer to the corresponding part of this project for the detailed installation instruction. 
