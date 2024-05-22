#!/bin/bash



python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aqfl  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-aqfl/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aql  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-aql/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-cell  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-cell/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-lrml  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-lrml/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-pfl  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-pfl/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-ipo  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/anonymous/zephyr-7b-g-ipo/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id hfh4-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --alpaca-model ./alpaca_eval/models_configs/HuggingFaceH4/zephyr-7b-g-dpo/configs.yaml \
    --alpaca-reference-model ./alpaca_eval/models_configs/HuggingFaceH4/gemma-7b-zephyr-sft/configs.yaml \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


######## Alpaca Eval 2.0 from Outputs ##############################

conda activate handbook_alpaca

export IS_ALPACA_EVAL_2=True

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/model_outputs.json \
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
    --model-id anonymous-zephyr-7b-g-padll  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-padll/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-dbaql  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dbaql/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aqfl  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-aqfl/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aql  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-aql/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-cell  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-cell/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-lrml  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-lrml/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-pfl  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-pfl/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id hfh4-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/hfh4-zephyr-7b-g-dpo/model_outputs.json \
    --alpaca-reference-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/reference_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


    ################################ Compare against GPT Turbo:

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dpo/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-hinge  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-hinge/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-padll  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-padll/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-dbaql  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-dbaql/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml


python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aqfl  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-aqfl/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-aql  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-aql/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-cell  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-cell/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-lrml  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-lrml/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id anonymous-zephyr-7b-g-pfl  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/anonymous-zephyr-7b-g-pfl/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml

python scripts/run_alpaca_eval.py \
    --model-id hfh4-zephyr-7b-g-dpo  \
    --num-generations 1 \
    --alpaca-eval \
    --from-outputs \
    --alpaca-model ./alpaca_eval/results/hfh4-zephyr-7b-g-dpo/model_outputs.json \
    --alpaca-openai-configs ./alpaca_eval/client_configs/openai_configs_single.yaml