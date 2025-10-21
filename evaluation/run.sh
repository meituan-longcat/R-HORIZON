
INPUT_FILE=$1
OUTPUT_DIR=$2
MODEL_NAME=$3
INPUT_FILENAME=$(basename "$INPUT_FILE" .jsonl)

mkdir -p $OUTPUT_DIR

# Step1 Inference
python evaluation/inference.py --input $INPUT_FILE --output $OUTPUT_DIR/${INPUT_FILENAME}_result.json --model_name $MODEL_NAME
# # Step2 use llm to extract nested answer from inference result
python evaluation/extract.py --input $OUTPUT_DIR/${INPUT_FILENAME}_result.json --output $OUTPUT_DIR/${INPUT_FILENAME}_result_judged.json --model_name gpt-4.1
# # Step3 use verify script to judge
python evaluation/judge.py --raw_input $INPUT_FILE --prediction $OUTPUT_DIR/${INPUT_FILENAME}_result_judged.json --output $OUTPUT_DIR/${INPUT_FILENAME}_result_judged_stat.txt