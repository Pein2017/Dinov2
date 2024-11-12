#!/bin/bash

# Set strict error handling
set -e
set -u
set -o pipefail

# Enable job control and trap Ctrl+C and SIGTERM
set -m
trap 'echo "Caught SIGINT or SIGTERM, terminating..."; kill -- -$$; exit 1' INT TERM

# Define variables
PROJECT_DIR="/data/training_code/Pein/dinov2"
GPUS_PER_NODE=8  

# Navigate to the project root directory
CONFIG_FILE="dinov2/configs/train/vits14.yaml"
TEMP_LOG_DIR="bbu_logs/bbu-vitb14-bs_pergpu_128"
NO_RESUME=true

PYTHON_SCRIPT="dinov2/train/train.py"
OUTPUT_DIR="$TEMP_LOG_DIR"

# Define a new port to avoid address in use error
PORT=29502  

# Print debug information
echo "Current directory: $PROJECT_DIR"
echo "Config file exists: $(test -f "$CONFIG_FILE" && echo 'Yes' || echo 'No')"

# If config file does not exist, stop or raise error
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi  

# Create necessary directories
mkdir -p "$TEMP_LOG_DIR"

# Set up Python path
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Training BBU model"
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

# Launch torchrun with the new master port
torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$PORT \
    "$PYTHON_SCRIPT" \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    ${TORCHRUN_ARGS:+"$TORCHRUN_ARGS"} \
    2>&1 | tee "$TEMP_LOG_DIR/training.log"

# Calculate training time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))

echo "Training completed"
echo "Total training time: ${hours}:${minutes} (hours:minutes)"