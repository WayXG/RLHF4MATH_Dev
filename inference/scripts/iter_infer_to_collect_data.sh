# Need to register the model first ... 

MODEL_NAME_OR_PATH="llm-agents/tora-code-34b-v1.0"

# DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

#DATA_NAME="math"
DATA_NAME="gsm8k"
OUTPUT_DIR="./output1"

SPLIT="train"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

for ((i=0; i<=8000; i+=1))
do
TOKENIZERS_PARALLELISM=false \
python -um data_collect.infer_api_collect_data \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed ${i} \
--temperature 1.0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 
done
