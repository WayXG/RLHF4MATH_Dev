DATA_NAME="gsm8k"
OUTPUT_DIR="./output1"

SPLIT="train"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=100
MODEL_NAME_OR_PATH=gemma

TOKENIZERS_PARALLELISM=false \
python -um data_collect.infer_api_collect_data \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 0 \
--temperature 1.0 \
--n_sampling 1 \
--top_p 1 \                                      
