#!/bin/bash

# Define common variables for input and ground truth files
INPUT_FOLDER="/home/davidchan/Projects/visual-puzzles/rebus_puzzle/image"
GROUND_TRUTH="/home/davidchan/Projects/visual-puzzles/rebus_puzzle/answers.json"

echo "Starting CLIP recall computations..."
echo "Input Folder: ${INPUT_FOLDER}"
echo "Ground Truth File: ${GROUND_TRUTH}"
echo "------------------------------------"

# 1. OpenAI's original CLIP ViT-B-32
echo "Running 1. OpenAI's original CLIP ViT-B-32..."
python compute_clip_recall.py \
  --checkpoint openai \
  --model ViT-B-32 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_vit_b32_openai.json
echo "------------------------------------"

# 2. LAION ViT-B-16 trained on LAION-400M
echo "Running 2. LAION ViT-B-16 trained on LAION-400M..."
python compute_clip_recall.py \
  --checkpoint laion400m_e31 \
  --model ViT-B-16 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_vit_b16_laion400m.json
echo "------------------------------------"

# 3. SigLIP ViT-SO400M-14
echo "Running 3. SigLIP ViT-SO400M-14..."
python compute_clip_recall.py \
  --checkpoint webli \
  --model ViT-SO400M-14-SigLIP-384 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_siglip_so400m.json
echo "------------------------------------"

# 4. MetaCLIP ViT-L-14
echo "Running 4. MetaCLIP ViT-L-14..."
python compute_clip_recall.py \
  --checkpoint metaclip_400m \
  --model ViT-L-14 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_metaclip_vitl14.json
echo "------------------------------------"

# 5. CoCa-ViT-L-14 finetuned on MSCOCO
echo "Running 5. CoCa-ViT-L-14 finetuned on MSCOCO..."
python compute_clip_recall.py \
  --checkpoint mscoco_finetuned_laion2b_s13b_b90k \
  --model coca_ViT-L-14 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_coca_vitl14_coco.json
echo "------------------------------------"

# 6. MobileCLIP-S2 (lightweight mobile model)
echo "Running 6. MobileCLIP-S2 (lightweight mobile model)..."
python compute_clip_recall.py \
  --checkpoint datacompdr \
  --model MobileCLIP-S2 \
  --input_folder "${INPUT_FOLDER}" \
  --ground_truth "${GROUND_TRUTH}" \
  --output ./metrics/metrics_mobileclip_s2.json
echo "------------------------------------"

echo "Starting SigLIP 2 model variants..."

# List of SigLIP 2 model variants and their corresponding '--model' argument values.
# The provided list was ambiguous, so the model names are parsed assuming the format
# follows the convention of previous SigLIP models where resolution is part of the name.
siglip2_models=(
  "ViT-B-16-SigLIP2-224"
  "ViT-B-16-SigLIP2-256"
  "ViT-B-16-SigLIP2-384"
  "ViT-B-16-SigLIP2-512"
  "ViT-L-16-SigLIP2-256"
  "ViT-L-16-SigLIP2-384"
  "ViT-L-16-SigLIP2-512"
  "ViT-SO400M-14-SigLIP2-384"
  "ViT-SO400M-14-SigLIP2-378"
  "ViT-SO400M-16-SigLIP2-256"
  "ViT-SO400M-16-SigLIP2-384"
  "ViT-SO400M-16-SigLIP2-512"
  "ViT-gopt-16-SigLIP2-256"
  "ViT-gopt-16-SigLIP2-384"
)

# Iterate through each SigLIP 2 model and run the computation
for model_name in "${siglip2_models[@]}"; do
  # Generate a unique output filename for each model
  # Converts model name to lowercase and replaces hyphens with underscores
  output_filename=$(echo "${model_name}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  output_path="./metrics/metrics_${output_filename}.json"

  echo "Running SigLIP 2 model: ${model_name}..."
  python compute_clip_recall.py \
    --checkpoint webli \
    --model "${model_name}" \
    --input_folder "${INPUT_FOLDER}" \
    --ground_truth "${GROUND_TRUTH}" \
    --output "${output_path}"
  echo "------------------------------------"
done

echo "All CLIP recall computations complete."
echo "Results are saved in JSON files in the current directory."
