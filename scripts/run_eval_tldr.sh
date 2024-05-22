#!/bin/bash

conda activate handbook_alpaca

export OPENAI_CLIENT_CONFIG_PATH="./alpaca_eval/client_configs/openai_configs_single.yaml"

########## Generate outputs and evaluate ########

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-sft/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/sft \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-dpo/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/dpo \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-padll/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/padll \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-hinge/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/hinge \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-aql/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/aql \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-lrml/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/lrml \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-cell/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/cell \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-pfl/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/pfl \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-dbaql/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/dbaql \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate_from_model \
    --model_configs ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-tldr-aqfl/configs.yaml \
    --output_path ./alpaca_eval/tldr_eval/aqfl \
    --evaluation_dataset ./alpaca_eval/tldr_eval/tldr_dataset.csv \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml


# ######### Evaluate against SFT from existing outputs

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/dpo/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/dpo/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml \

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/hinge/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/hinge/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/aql/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/aql/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/padll/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/padll/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/cell/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/cell/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/lrml/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/lrml/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/pfl/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/pfl/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/dbaql/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/dbaql/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml

alpaca_eval evaluate \
    --model_outputs ./alpaca_eval/tldr_eval/aqfl/model_outputs.json \
    --reference_outputs ./alpaca_eval/tldr_eval/sft/model_outputs.json \
    --output_path ./alpaca_eval/tldr_eval/aqfl/sft \
    --annotators_config ./alpaca_eval/evaluator_configs/tldr_concise/configs.yaml