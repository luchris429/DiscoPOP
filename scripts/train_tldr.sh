#!/bin/bash

source activate handbook

CONFIG_FILE="recipes/accelerate_configs/deepspeed_zero3.yaml"
SCRIPT="scripts/run_gpo_tldr.py"
CONFIG_YAML="recipes/zephyr-7b-gemma/gpo/config_tldr.yaml"
GRADIENT_ACCUMULATION_STEPS=16

declare -A LOSS_TYPES
LOSS_TYPES=(
    ["dpo"]="data/zephyr-7b-gemma-dpo-tldr-small-corrected-2"
    # ["hinge"]="data/zephyr-7b-gemma-hinge-tldr-small-corrected"
    # ["kto_pair"]="data/zephyr-7b-gemma-kto-tldr-small-corrected"
    # ["dbaql"]="data/zephyr-7b-gemma-dbaql-tldr-small-corrected"
    # ["aql"]="data/zephyr-7b-gemma-aql-tldr-small-corrected"
    # ["padll"]="data/zephyr-7b-gemma-padll-tldr-small-corrected"
    # ["aqfl"]="data/zephyr-7b-gemma-aqfl-tldr-small-corrected"
    # ["cell"]="data/zephyr-7b-gemma-cell-tldr-small-corrected"
    # ["lrml"]="data/zephyr-7b-gemma-lrml-tldr-small-corrected"
    # ["pfl"]="data/zephyr-7b-gemma-pfl-tldr-small-corrected"
)

for LOSS_TYPE in "${!LOSS_TYPES[@]}"; do
  OUTPUT_DIR="${LOSS_TYPES[$LOSS_TYPE]}"
  accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes="4" \
    $SCRIPT $CONFIG_YAML \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --loss_type="$LOSS_TYPE" \
    --output_dir="$OUTPUT_DIR"
done