#!/bin/bash

# Activate conda environment
conda activate handbook

# Function to run the alpaca_eval script
run_alpaca_eval() {
    local model_id=$1
    local model_path=$2
    local ref_model_path=$3
    local config_path=$4
    local from_outputs=$5

    if [ "$from_outputs" = true ]; then
        python scripts/run_alpaca_eval.py \
            --model-id "$model_id" \
            --num-generations 1 \
            --alpaca-eval \
            --from-outputs \
            --alpaca-model "$model_path" \
            --alpaca-reference-model "$ref_model_path" \
            --alpaca-openai-configs "$config_path"
    else
        python scripts/run_alpaca_eval.py \
            --model-id "$model_id" \
            --num-generations 1 \
            --alpaca-eval \
            --alpaca-model "$model_path" \
            --alpaca-reference-model "$ref_model_path" \
            --alpaca-openai-configs "$config_path"
    fi
}

# Directories and configurations
base_dir="./alpaca_eval"
model_dir="$base_dir/models_configs"
result_dir="$base_dir/results"
config_path="$base_dir/client_configs/openai_configs_single.yaml"
ref_model_path="$model_dir/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml"
ref_model_output="$result_dir/anonymous-zephyr-7b-g-dpo/reference_outputs.json"

# Models to evaluate
declare -A models=(
    [hfh4-zephyr-7b-g-dpo]="$model_dir/HuggingFaceH4/zephyr-7b-g-dpo/configs.yaml"
    [anonymous-zephyr-7b-g-dpo]="$model_dir/anonymous/zephyr-7b-g-dpo/configs.yaml"
    [anonymous-zephyr-7b-g-hinge]="$model_dir/anonymous/zephyr-7b-g-hinge/configs.yaml"
    [anonymous-zephyr-7b-g-kto]="$model_dir/anonymous/zephyr-7b-g-kto/configs.yaml"
    [anonymous-zephyr-7b-g-dbaql]="$model_dir/anonymous/zephyr-7b-g-dbaql/configs.yaml"
    [anonymous-zephyr-7b-g-aql]="$model_dir/anonymous/zephyr-7b-g-aql/configs.yaml"
    [anonymous-zephyr-7b-g-padll]="$model_dir/anonymous/zephyr-7b-g-padll/configs.yaml"
    [anonymous-zephyr-7b-g-aqfl]="$model_dir/anonymous/zephyr-7b-g-aqfl/configs.yaml"
    [anonymous-zephyr-7b-g-cell]="$model_dir/anonymous/zephyr-7b-g-cell/configs.yaml"
    [anonymous-zephyr-7b-g-lrml]="$model_dir/anonymous/zephyr-7b-g-lrml/configs.yaml"
    [anonymous-zephyr-7b-g-pfl]="$model_dir/anonymous/zephyr-7b-g-pfl/configs.yaml"
)

# Run evaluations without --from-outputs --> this creates the generations
for model_id in "${!models[@]}"; do
    run_alpaca_eval "$model_id" "${models[$model_id]}" "$ref_model_path" "$config_path" false
done


conda activate handbook_alpaca
export IS_ALPACA_EVAL_2=True

# Models to evaluate with --from-outputs --> Here the outputs are compared always with the same reference output
declare -A models_from_outputs=(
    [hfh4-zephyr-7b-g-dpo]="$result_dir/hfh4-zephyr-7b-g-dpo/model_outputs.json"
    [anonymous-zephyr-7b-g-dpo]="$result_dir/anonymous-zephyr-7b-g-dpo/model_outputs.json"
    [anonymous-zephyr-7b-g-hinge]="$result_dir/anonymous-zephyr-7b-g-hinge/model_outputs.json"
    [anonymous-zephyr-7b-g-kto]="$result_dir/anonymous-zephyr-7b-g-kto/model_outputs.json"
    [anonymous-zephyr-7b-g-padll]="$result_dir/anonymous-zephyr-7b-g-padll/model_outputs.json"
    [anonymous-zephyr-7b-g-dbaql]="$result_dir/anonymous-zephyr-7b-g-dbaql/model_outputs.json"
    [anonymous-zephyr-7b-g-aqfl]="$result_dir/anonymous-zephyr-7b-g-aqfl/model_outputs.json"
    [anonymous-zephyr-7b-g-aql]="$result_dir/anonymous-zephyr-7b-g-aql/model_outputs.json"
    [anonymous-zephyr-7b-g-cell]="$result_dir/anonymous-zephyr-7b-g-cell/model_outputs.json"
    [anonymous-zephyr-7b-g-lrml]="$result_dir/anonymous-zephyr-7b-g-lrml/model_outputs.json"
    [anonymous-zephyr-7b-g-pfl]="$result_dir/anonymous-zephyr-7b-g-pfl/model_outputs.json"
)

# Run evaluations with --from-outputs
for model_id in "${!models_from_outputs[@]}"; do
    run_alpaca_eval "$model_id" "${models_from_outputs[$model_id]}" "$ref_model_output" "$config_path" true
done

# Compare against GPT Turbo
declare -a compare_models=(
    hfh4-zephyr-7b-g-dpo
    anonymous-zephyr-7b-g-dpo
    anonymous-zephyr-7b-g-hinge
    anonymous-zephyr-7b-g-kto
    anonymous-zephyr-7b-g-padll
    anonymous-zephyr-7b-g-dbaql
    anonymous-zephyr-7b-g-aqfl
    anonymous-zephyr-7b-g-aql
    anonymous-zephyr-7b-g-cell
    anonymous-zephyr-7b-g-lrml
    anonymous-zephyr-7b-g-pfl
)

for model_id in "${compare_models[@]}"; do
    run_alpaca_eval "$model_id" "$result_dir/$model_id/model_outputs.json" "" "$config_path" true
done
