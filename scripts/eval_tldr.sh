#!/bin/bash

conda activate handbook_alpaca

export OPENAI_CLIENT_CONFIG_PATH="./alpaca_eval/client_configs/openai_configs_single.yaml"

# Declare associative arrays for model configs and output paths
declare -A MODEL_CONFIGS
declare -A OUTPUT_PATHS

MODEL_CONFIGS=(
    ["sft"]="sft"
    ["dpo"]="dpo"
    ["hinge"]="hinge"
    ["kto"]="kto"
    ["dbaql"]="dbaql"
    ["aql"]="aql"
    ["padll"]="padll"
    ["aqfl"]="aqfl"
    ["cell"]="cell"
    ["lrml"]="lrml"
    ["pfl"]="pfl"
)


EVALUATION_DATASET="./alpaca_eval/tldr_eval/tldr_dataset.csv"
ANNOTATORS_CONFIG="./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml"

# Generate outputs and evaluate
for KEY in "${!MODEL_CONFIGS[@]}"; do
    alpaca_eval evaluate_from_model \
        --model_configs "./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-${MODEL_CONFIGS[$KEY]}/configs.yaml" \
        --output_path "./alpaca_eval/tldr_eval/${MODEL_CONFIGS[$KEY]}" \
        --evaluation_dataset $EVALUATION_DATASET \
        --annotators_config $ANNOTATORS_CONFIG
done

# Evaluate against SFT from existing outputs
for KEY in "${!MODEL_CONFIGS[@]}"; do
    if [[ "$KEY" != "sft" ]]; then
        alpaca_eval evaluate \
            --model_outputs "./alpaca_eval/tldr_eval/${MODEL_CONFIGS[$KEY]}/model_outputs.json" \
            --reference_outputs "./alpaca_eval/tldr_eval/sft/model_outputs.json" \
            --output_path "./alpaca_eval/tldr_eval/${MODEL_CONFIGS[$KEY]}/sft" \
            --annotators_config $ANNOTATORS_CONFIG
    fi
done