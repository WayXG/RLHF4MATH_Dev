# RLHF4MATH_Dev

TL;DL: this is an implementation of the Online Iterative Direct Preference Learning for Multi-turn Mathematical Reasoning with External Tools.

We consider the math problem solving with python interpreter, which means that the model can write a python code and ask the external environmnet to run and receive the excutaion result, before the LLM makes its next decision.

## Installation instructions

The main pipeline is divided into three steps:

- Supervised Fine-tuning (SFT);
- Data generation and annotation;
- Multi-turn DPO/KTO Training.

It is recommended to have three separate environments for **sft**, **inference**, and **dpo_train**. Please refer to the corresponding part of this project for the detailed installation instruction. 
