# KTO Training with Multi-turn Data


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

**Hack DPO Trainer**

Now we need to hack the DPO trainer to implement some additional functions. 

We highlight the modified part with ############## MODIFICATION.

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/dpo_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm dpo_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv dpo_train/dpo_trainer.py anaconda3/envs/dpo_train/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py
```

## Running the Code

Running the code before modify num_processes: 8 in ./training_configs/zero2_pf.yaml, the number 8 means that you will use 8 GPUs. Also modify the parameters, models, and datasets provided in run_dpo.py.

```shell
accelerate launch --config_file zero2_for_dpo.yaml run_dpo.py 
```

If you encounter out-of-memory issue. Running the code with Gemma-7b-it with zero3_pf.yaml. You can also reduce the max length of the data.


## Modification 

