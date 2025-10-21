#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return the exit status of the last command in the pipe that failed.

# ========================================================================================
# R-HORIZON: Distributed Reinforcement Learning Training Script
#
# This script launches a distributed training job using Ray and the GRPO algorithm.
# It is designed to be run in a multi-GPU, multi-node environment.
#
# Usage:
#   - For single-node training: bash scripts/train_rl.sh
#   - For multi-node training: Ensure the Ray cluster is initialized, then run this
#     script on each node.
#   - Customize parameters by passing them as arguments, e.g.,
#     bash scripts/train_rl.sh --model_path "/path/to/your/model" --output_dir "/path/to/save"
# ========================================================================================

# ---
# ⚙️ Default Configuration
# ---
# These can be overridden by command-line arguments.
MODEL_PATH="r1-qwen-7b" # Path to your base model
TRAIN_DATA_DIR="./training/data/" # Directory for training data
EVAL_DATA_DIR="./training/data/"   # Directory for evaluation data
OUTPUT_DIR="./checkpoints/r-horizon-rl-training" # Directory to save checkpoints and logs

WORLD_SIZE=1  # Number of nodes
GPUS_PER_NODE=8 # Number of GPUs per node
MASTER_PORT=29500

# ---
# ↗️ Command-line Argument Parsing
# ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --train_data_dir) TRAIN_DATA_DIR="$2"; shift ;;
        --eval_data_dir) EVAL_DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --world_size) WORLD_SIZE="$2"; shift ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift ;;
        --master_port) MASTER_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ---
# 🛠️ Environment Setup
# ---
# Set PyTorch and NCCL environment variables for performance and debugging.
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export NCCL_DEBUG="WARN"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_ATTENTION_BACKEND="XFORMERS"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# ---
# 🚀 Training Hyperparameters
# ---
# Rollout and PPO settings
ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096 # Renamed from RES_LENGTH for clarity
GROUP_SIZE=16
N_VAL_SAMPLES=8
TRAIN_TEMPERATURE=1.0

# Tensor/Sequence Parallelism (for very large models)
TP=1 # Tensor Parallelism
SP=1 # Sequence Parallelism
MAX_TOKEN_LEN=$(((RESPONSE_LENGTH + MAX_PROMPT_LENGTH + 1000) / SP))

# ---
# 📊 Dataset Configuration
# ---
# Assumes data files are in the specified directories.
# Modify the file names if your dataset structure is different.
train_files="[\"$TRAIN_DATA_DIR/combined_key_var_k2_sd43_passrate0.25_skywork_or1_train_7b_math_key_variables_filtered.pkl\"]"
test_files="[\"$EVAL_DATA_DIR/aime24.parquet\",\"$EVAL_DATA_DIR/aime25.parquet\"]"

# ---
#  wandb Configuration (optional)
# ---
# Set to "online" to enable Weights & Biases logging.
# Ensure you have run `wandb login` first.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"

# ---
# Ray Cluster & Job Submission
# ---
# This script assumes a Ray cluster is already running. The head node address
# should be discoverable. For a single-node setup, it starts a temporary cluster.
if [[ "$WORLD_SIZE" -eq 1 ]]; then
    echo "Starting a single-node Ray cluster..."
    ray start --head --port=$MASTER_PORT --num-gpus=$GPUS_PER_NODE --include-dashboard=false
    RAY_ADDRESS="127.0.0.1:$MASTER_PORT"
else
    # For multi-node, please start the Ray cluster manually following Ray documentation.
    # The head node address must be set in the RAY_ADDRESS environment variable.
    if [[ -z "${RAY_ADDRESS}" ]]; then
        echo "Error: RAY_ADDRESS environment variable is not set for multi-node training."
        echo "Please set it to the address of the Ray head node (e.g., export RAY_ADDRESS='<head_node_ip>:6379')."
        exit 1
    fi
    echo "Connecting to existing Ray cluster at $RAY_ADDRESS"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
STATS_DIR="${OUTPUT_DIR}/stats"
mkdir -p "$STATS_DIR"

# Project and Experiment Naming
PROJECT_NAME="R-HORIZON-RL-Training"
EXP_NAME="grpo-$(basename ${MODEL_PATH})-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting Ray job..."
echo "  - Model: ${MODEL_PATH}"
echo "  - Output Dir: ${OUTPUT_DIR}"
echo "  - Train Data: ${train_files}"
echo "  - Eval Data: ${test_files}"

# Submit the training job to the Ray cluster.
# The entry point is assumed to be `verl.trainer.main_ppo`.
# The configuration is passed using Hydra-style overrides.
ray job submit \
    --address "$RAY_ADDRESS" \
    --runtime-env="./runtime_env.yaml" \
    --working-dir="." \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_reward_clip=True \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.n_val=$N_VAL_SAMPLES \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=10 \
    trainer.stats_path=$STATS_DIR \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=30

echo "✅ Training job submitted."

# Stop the local Ray cluster if it was started by this script
if [[ "$WORLD_SIZE" -eq 1 ]]; then
    ray stop
fi