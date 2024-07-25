# RLHF4MATH_Dev

TL;DL: this is an implementation of the Online Iterative Direct Preference Learning for Multi-turn Mathematical Reasoning with External Tools.

We consider the math problem solving with python interpreter, which means that the model can write a python code and ask the external environmnet to execute and receive the excutaion result, before the LLM makes its next decision.

## Installation instructions

The main pipeline is divided into three steps:

- Supervised Fine-tuning (SFT);
- Data generation and annotation;
- Multi-turn DPO/KTO Training.

It is recommended to have three separate environments for **sft**, **inference**, and **dpo_train**. Please refer to the corresponding part of this project for the detailed installation instruction. 


## Citation

If you find the content of this repo useful, please consider cite it as follows:

```bibtex
@misc{dong2024rlhf,
      title={Online Iterative Direct Preference Learning for Multi-turn Mathematical Reasoning with External Tools}, 
      author={XX},
      year={2024},
      eprint={2405.07863},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
