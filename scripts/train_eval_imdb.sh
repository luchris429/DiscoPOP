#!/bin/bash

source activate handbook

CONFIG_FILE="recipes/accelerate_configs/deepspeed_zero3.yaml"
NUM_PROCESSES=4
SCRIPT="scripts/run_gpo_imdb.py"
CONFIG_YAML="recipes/zephyr-7b-gemma/gpo/config_imdb.yaml"
GRADIENT_ACCUMULATION_STEPS=16

BETAS=(
  # 5.0 
  # 2.5 
  1.0 
  0.5 
  0.25 
  0.1 
  0.05 
  0.025 
  # 0.01 
)

LOSS_TYPES=(
    # "dpo"
    # "hinge"
    "kto"
    # "dbaql"
    # "aql"
    # "padll"
    # "aqfl"
    # "cell"
    # "lrml"
    # "pfl"
)

for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
  for BETA in "${BETAS[@]}"; do
    OUTPUT_DIR="data/imdb_redone/gpt-2-${LOSS_TYPE}-imdb-20240511-beta-${BETA//.}"
    accelerate launch \
      --config_file "$CONFIG_FILE" \
      --num_processes="$NUM_PROCESSES" \
      "$SCRIPT" "$CONFIG_YAML" \
      --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
      --loss_type="$LOSS_TYPE" \
      --beta="$BETA" \
      --output_dir="$OUTPUT_DIR" 
    python scripts/run_eval_imdb.py \
      --loss="$LOSS_TYPE" \
      --beta="$BETA" \
      --checkpoint="$OUTPUT_DIR"
  done
done