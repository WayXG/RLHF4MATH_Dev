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

**Hack DPO Trainer**

Now we need to hack the DPO trainer to implement some additional functions. See **Modification** below for details. 

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/dpo_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm dpo_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv dpo_train/dpo_trainer.py anaconda3/envs/dpo_train/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py
```

## Running the Code

Running the code with Gemma.

```shell
accelerate launch --config_file zero2_for_dpo.yaml run_dpo.py 
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the run_dpo.py 

If you encounter out-of-memory issue. Running the code with Gemma-2b-it with zero2_for_dpo.yaml or reduce the max length of the data.


## Modification 

Some of the modifications are currently hard coded in the codes and should be fixed later. We summarize them here. For all the modifications, you can search by ``HARD CODE'' to find them. 

### User Turn Mask

In dpo.py, we will detect all the user turn and set the label to be $-100$ so that the DPO trainer will ignore them. The current implementation is specialized for Gemma as we simply detect the chat template of Gemma: [106, 1645, 108], and [107, 108], corresponding to <star_of_turn>, <end_of_turn>...

```python
############## HARD CODE

def get_new(f, old_labels):
    # We mask the user turn to create new labels
    labels = copy.deepcopy(old_labels)
    #masks = copy.deepcopy(old_masks)
    start = False
    for j in range(len(f)):
        
        if f[j:j+3] == [106, 1645, 108]:
            start = True
        if f[j:j+2] == [107, 108] and start:
            labels[j] = -100
            labels[j+1] = -100
            #masks[j] = 0
            #masks[j+1] = 0
            start = False
        if start:
            labels[j] = -100
            #masks[j] = 0
    return labels
############
```

- dpo.py: function get_new
- dpo.py: tokenize_batch_element

### Additional NLL Loss

We additionally add an NLL loss into the dpo training. To do this, we hack the dpo_loss function in dpo.py.

```python
############## HARD CODE
losses = losses + 1.0 * policy_nll_loss # change the coefficient of NLL loss here.
#############
```

where the NLL loss in computed in the hacked dpo_trainer.py. 


```python
############## HARD CODE
def cross_entropy_loss(logits, labels):
    if not self.is_encoder_decoder:
        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)
    return loss

labels = concatenated_batch["concatenated_labels"].clone()
nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])
#########################################
```


