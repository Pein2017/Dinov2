#!/bin/bash

# Define a default value to avoid unbound variable errors
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU}


# Set strict error handling
set -e
set -u
set -o pipefail

# Enable job control and trap Ctrl+C and SIGTERM
set -m
trap 'echo "Caught SIGINT or SIGTERM, terminating..."; kill -- -$$; exit 1' INT TERM

# Define variables
PROJECT_DIR="/data/training_code/Pein/dinov2"
GPUS_PER_NODE=6
# Add a variable to manually set CUDA devices. Set to "" to auto-detect.
MANUAL_GPU_LIST=0,1,2,3,4,5

cd "$PROJECT_DIR"

# Navigate to the project root directory
CONFIG_FILE=dinov2/configs/train/vits14_pretrained.yaml

# Define root and experiment names based on hyperparameters
ROOT_LOG_DIR=joined_logs
EXPERIMENT_NAME=vits14-total_bs_384-lr_1e-4-epochs_100-epoch_len_2500-warmup_10-teacher_warmup_30-pretrained
LOG_DIR="$ROOT_LOG_DIR/$EXPERIMENT_NAME"

NO_RESUME=true

PYTHON_SCRIPT=dinov2/train/train.py
OUTPUT_DIR="$LOG_DIR"

# Define a new port to avoid address in use error
PORT=29501

# Print debug information
echo "Current directory: $PROJECT_DIR"
echo "Config file exists: $(test -f "$CONFIG_FILE" && echo 'Yes' || echo 'No')"

# If config file does not exist, stop or raise error
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi  

# Create necessary directories
mkdir -p "$LOG_DIR"

# Set up Python path
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Training JOINED model"
echo "Updated directory: $PROJECT_DIR"
echo "Updated Python path: $PYTHONPATH"

# Record start time
start_time=$(date +%s)

# Initialize TORCHRUN_ARGS as empty
TORCHRUN_ARGS=""

# Handle --no-resume flag
if [ "$NO_RESUME" = true ]; then
    TORCHRUN_ARGS="--no-resume"
fi

# Initialize conda environment
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate openmm
set -u

# Function to find available GPUs
get_available_gpus() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    awk '$1 < 1000 {print NR-1}' | paste -sd "," -
}

# Modify GPU selection logic to use manual list if provided
if [ -n "$MANUAL_GPU_LIST" ]; then
    AVAILABLE_GPUS="$MANUAL_GPU_LIST"
    GPUS_PER_NODE=$(echo "$AVAILABLE_GPUS" | tr -cd ',' | wc -c)
    GPUS_PER_NODE=$((GPUS_PER_NODE + 1))
else
    AVAILABLE_GPUS=$(get_available_gpus)
    GPUS_PER_NODE=$(echo "$AVAILABLE_GPUS" | tr -cd ',' | wc -c)
    GPUS_PER_NODE=$((GPUS_PER_NODE + 1))
fi

export CUDA_VISIBLE_DEVICES="$AVAILABLE_GPUS"

# Launch torchrun with the new master port
torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$PORT \
    "$PYTHON_SCRIPT" \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    ${TORCHRUN_ARGS:+"$TORCHRUN_ARGS"} \
    2>&1 | tee "$LOG_DIR/training.log"

# Calculate training time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))

echo "Training completed"
echo "Total training time: ${hours}:${minutes} (hours:minutes)"