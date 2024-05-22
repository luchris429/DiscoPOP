#!/bin/bash

# Gemma Zephyr 2B with 256 Tokens

# conda activate handbook

# python scripts/run_mt_bench.py \
#     --model-id anonymous-zephyr-7b-g-dpo \
#     --model-path anonymous/zephyr-7b-gemma-dpo \
#     --num-generations 20 \
#     --mt-bench \
#     --max-new-token 1024

# conda activate handbook

# python scripts/run_mt_bench.py \
#     --model-id hfh4-zephyr-7b-g-dpo-variance \
#     --model-path HuggingFaceH4/zephyr-7b-gemma-v0.1 \
#     --num-generations 10 \
#     --mt-bench \
#     --max-new-token 1024

# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-dpo  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-dpo/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id hfh4-zephyr-7b-g-dpo  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/HuggingFaceH4/zephyr-7b-g-dpo/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-hinge  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-hinge/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml


# This is the PADLL model
# conda activate handbook

# python scripts/run_mt_bench.py \
#     --model-id anonymous-zephyr-7b-g-padll \
#     --model-path anonymous/zephyr-7b-gemma-performance_adaptive_decay_logistic_loss \
#     --num-generations 1 \
#     --mt-bench \
#     --max-new-token 1024

# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-padll  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-padll/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml


# BLine

# python scripts/run_mt_bench.py \
#     --model-id anonymous-zephyr-7b-g-bline \
#     --model-path anonymous/zephyr-7b-gemma-bline \
#     --num-generations 1 \
#     --mt-bench \
#     --max-new-token 1024

# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-bline  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-bline/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml


# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-hinge  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-hinge/configs.yaml \
#     --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml



############## Eval from ouputs #################
# conda activate handbook_alpaca

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id hfh4-zephyr-7b-g-dpo  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --from-outputs \
#     --alpaca-model ./alpaca_eval/results/hfh4-zephyr-7b-g-dpo/model_outputs.json \
#     --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-padll  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --from-outputs \
#     --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-padll/model_outputs.json \
#     --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-bline  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --from-outputs \
#     --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-bline/model_outputs.json \
#     --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-hinge  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --from-outputs \
#     --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-hinge/model_outputs.json \
#     --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml

# export IS_ALPACA_EVAL_2=False
# python scripts/run_alpaca_eval.py \
#     --model-id anonymous-zephyr-7b-g-dpo  \
#     --num-generations 1 \
#     --alpaca-eval \
#     --from-outputs \
#     --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/model_outputs.json \
#     --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
#     --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_multi.yaml


######## Alpaca Eval 2.0 from Outputs ##############################

conda activate handbook_alpaca

#export IS_ALPACA_EVAL_2=True
python scripts/run_alpaca_eval.py \
    --model-id hfh4-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/hfh4-zephyr-7b-g-dpo/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-padll  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-padll/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-bline  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-bline/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-hinge  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-hinge/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml
