# DPO Training with Multi-turn Data


## Installation instructions

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


Before starting, please make sure your linux machine has nvidia-cuda-toolkit installed. See SFT part for the guidance. 


**Training Environment**

```sh
conda create -n dpo_train python=3.10.9
conda activate dpo_train

# The training environment is modified from the Zephyr project
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2
pip install numpy==1.26.4
# You also need to install the wandb to record the training and login with your huggingface account so that you have access to the Gemma models.

pip install wandb
wandb login
huggingface-cli login
```

## Running the Code

Running the code with Gemma.

```shell
accelerate launch --config_file zero2_for_dpo.yaml run_dpo.py 
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the run_dpo.py 

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with zero2_for_dpo.yaml or reduce the max length of the data.


## Modification 

Some of the modifications are currently hard coded in the codes and should be fixed later. We summarize them here.
