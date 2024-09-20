# DPO/KTO Training with Multi-turn Data


## 1 Installation instructions

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


Before starting, please make sure your linux machine has nvidia-cuda-toolkit installed. See SFT part for the guidance. 


**Training Environment**

```sh
conda create -n alignment_train python=3.10.9
conda activate alignment_train

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0

pip install wandb
wandb login
huggingface-cli login
```

## 2 Hakcing the DPO Trainer and KTO Trainer

### 2.1 Hack DPO Trainer

The code is based on RLHFlow/Online-RLHF but we need to hack the trainer to implement some additional functions. We highlight the modified part with ############## MODIFICATION.

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm dpo_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv dpo_train/dpo_trainer.py anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py
```

### 2.2 Hack KTO Trainer

The code is based on RLHFlow/Online-RLHF but we need to hack the KTO trainer to implement some additional functions. We highlight the modified part with ############## MODIFICATION.

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm kto_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv kto_train/kto_trainer.py anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/kto_trainer.py

# Step 4: modify the KTO config according to your GPU resource.
vim ./trl/trainer/kto_config.py
max_length: Optional[int] = 2048
max_prompt_length: Optional[int] = 1024
max_completion_length: Optional[int] = 2048
```

3 ## Running the Code

### 1 DPO
Running the code before modify num_processes: 8 in ./training_configs/zero2_pf.yaml, the number 8 means that you will use 8 GPUs. Also modify the parameters, models, and datasets provided in run_dpo.py.

```shell
accelerate launch --config_file ./training_configs/zero2_pf.yaml run_dpo.py ./training_configs/training.yaml

```

### 2 KTO 

```shell
bash run_kto.sh
```

If you encounter out-of-memory issue. Running the code with Gemma-7b-it with zero3_pf.yaml. You can also reduce the max length of the data.




