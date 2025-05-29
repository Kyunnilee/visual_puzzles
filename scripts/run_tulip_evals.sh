#!/bin/bash

# Define common variables
COMPUTE_SCRIPT="./compute_clip_recall.py"
INPUT_FOLDER="../dataset/image"
GROUND_TRUTH="../dataset/answers.json"
CHECKPOINTS_DIR="PATH_TO_TULIP_CHECKPOINTS/"

echo "Starting CLIP recall computations for Tulip checkpoints..."
echo "Input Folder: ${INPUT_FOLDER}"
echo "Ground Truth File: ${GROUND_TRUTH}"
echo "Checkpoints Directory: ${CHECKPOINTS_DIR}"
echo "------------------------------------"

# Checkpoint 1: tulip-B-16-224.ckpt
# Inferred Model: ViT-B-16-224
# CHECKPOINT_FILE="${CHECKPOINTS_DIR}tulip-B-16-224.ckpt"
# MODEL_NAME="TULIP-B-16-224"
# OUTPUT_FILE="./metrics_tulip_vit_b16_224.json"

# echo "Running for checkpoint: ${CHECKPOINT_FILE}, Model: ${MODEL_NAME}"
# python "${COMPUTE_SCRIPT}" \
#   --checkpoint "${CHECKPOINT_FILE}" \
#   --model "${MODEL_NAME}" \
#   --input_folder "${INPUT_FOLDER}" \
#   --ground_truth "${GROUND_TRUTH}" \
#   --output "${OUTPUT_FILE}"
# echo "------------------------------------"

# Checkpoint 2: tulip-G-16-384.ckpt
# Inferred Model: ViT-gopt-16-384 (assuming 'G' maps to 'gopt' as seen in SigLIP2 models)
CHECKPOINT_FILE="${CHECKPOINTS_DIR}tulip-G-16-384.ckpt"
MODEL_NAME="TULIP-G-16-384"
OUTPUT_FILE="./metrics/metrics_tulip_vit_gopt_16_384.json"

echo "Running for checkpoint: ${CHECKPOINT_FILE}, Model: ${MODEL_NAME}"
python "${COMPUTE_SCRIPT}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --model "${MODEL_NAME}" \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output "${OUTPUT_FILE}"
echo "------------------------------------"

# Checkpoint 3: tulip-so400m-14-384.ckpt
# Inferred Model: ViT-SO400M-14-384
CHECKPOINT_FILE="${CHECKPOINTS_DIR}tulip-so400m-14-384.ckpt"
MODEL_NAME="TULIP-so400m-14-384"
OUTPUT_FILE="./metrics/metrics_tulip_vit_so400m_14_384.json"

echo "Running for checkpoint: ${CHECKPOINT_FILE}, Model: ${MODEL_NAME}"
python "${COMPUTE_SCRIPT}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --model "${MODEL_NAME}" \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output "${OUTPUT_FILE}"
echo "------------------------------------"

echo "All specified Tulip checkpoint computations complete."
echo "Results are saved in JSON files in the current directory."
