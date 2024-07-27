if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
MODEL_NAME_OR_PATH=$1

#DATA_NAME="math"
DATA_NAME="gsm8k"
OUTPUT_DIR="./output1"

SPLIT="train"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

for ((i=0; i<=8000; i+=1))
do
TOKENIZERS_PARALLELISM=false \
python -um infer_data.collect_data \
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
