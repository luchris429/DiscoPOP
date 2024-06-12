import argparse
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd


def evaluate_gpo(model_id, alpaca_eval=False, alpaca_configs=None):
    alpaca_score = np.nan
    if alpaca_eval:
        print("Running Alpaca Eval")
        assert (
            alpaca_configs is not None
        ), "If you wan to evaluate on Alpaca Eval, you need to provide the alpaca configs"

        os.environ["OPENAI_CLIENT_CONFIG_PATH"] = alpaca_configs["openai_configs"]

        # Run Alpaca Eval script
        if alpaca_configs["from_outputs"]:
            if alpaca_configs["reference_model"] == "":
                command = [
                    "alpaca_eval",
                    "evaluate",
                    "--model_outputs",
                    alpaca_configs["model"],
                    "--output_path",
                    f"alpaca_eval/results_outputs_AE_gpt/{model_id}",
                ]
            else:
                command = [
                    "alpaca_eval",
                    "evaluate",
                    "--model_outputs",
                    alpaca_configs["model"],
                    "--reference_outputs",
                    alpaca_configs["reference_model"],
                    "--output_path",
                    f"alpaca_eval/results_from_outputs_AE2/{model_id}",
                ]

        else:
            command = [
                "alpaca_eval",
                "evaluate_from_model",
                "--model_configs",
                alpaca_configs["model"],
                "--reference_model_configs",
                alpaca_configs["reference_model"],
                "--output_path",
                f"alpaca_eval/results/{model_id}",
            ]
        result = subprocess.run(command)
        if result.returncode != 0:
            return (
                False,
                f"Alpaca Eval 2 failed with return code: {result.returncode}\n{result.stderr}",
            )

        if alpaca_configs["from_outputs"]:
            df_alpaca = pd.read_csv(
                f"alpaca_eval/results_from_outputs_AE2/{model_id}/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv",
                index_col="Unnamed: 0",
            )
        else:
            df_alpaca = pd.read_csv(
                f"alpaca_eval/results/{model_id}/alpaca_eval_gpt4/leaderboard.csv",
                index_col="Unnamed: 0",
            )
        print("\n########## ALPACA EVAL ##########")
        alpaca_score = df_alpaca.loc[model_id, "win_rate"]
        print(f"{df_alpaca.loc[model_id, 'win_rate']:.3f} +- {df_alpaca.loc[model_id, 'standard_error']:.3f}")

    return True, alpaca_score


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-generations",
    type=int,
    default=5,
    help="Number of times the evaluation is to run",
)
parser.add_argument(
    "--model-id",
    type=str,
    help="Your personal model ID, does not have to match the model path name",
)
parser.add_argument("--alpaca-eval", action="store_true", help="Whether to evaluate on Alpaca Eval 2.0")
parser.add_argument("--no-logging", action="store_true")
parser.add_argument("--csv-save-path", type=str, default="./", help="Path to save the CSV file")
parser.add_argument(
    "--alpaca-model",
    type=str,
    default="",
    help="Path to the config file of the model, for Alpaca Eval 2.0",
)
parser.add_argument(
    "--alpaca-reference-model",
    type=str,
    default="",
    help="Path to reference model, for Alpaca Eval 2.0",
)
parser.add_argument(
    "--alpaca-openai-configs",
    type=str,
    help="Path to OpenAI API config file, for Alpaca Eval 2.0",
)
parser.add_argument(
    "--from-outputs",
    action="store_true",
    help="Whether to evaluate alpaca from model or from outputs",
)


if __name__ == "__main__":
    args = parser.parse_args()

    now = str(datetime.now()).replace(" ", "_")
    if not args.no_logging:
        save_dir = f"runs/{now}"
        os.mkdir(save_dir)

    config = {
        "NUM_GENERATIONS": args.num_generations,
        "SAVE_DIR": save_dir,
        "NOW": now,
    }

    alpaca_configs = None
    if args.alpaca_eval:
        alpaca_configs = {
            "model": args.alpaca_model,
            "reference_model": args.alpaca_reference_model,
            "openai_configs": args.alpaca_openai_configs,
            "from_outputs": args.from_outputs,
        }

    alpaca_eval_values = []
    model_id = args.model_id

    # TODO: DONT HARD CODE PATHS
    csv_save_path = "."

    for i in range(args.num_generations):
        print(f"Iteration: {i}")
        evaluated, val = evaluate_gpo(model_id, alpaca_eval=args.alpaca_eval, alpaca_configs=alpaca_configs)
        if evaluated:
            alpaca_eval_values.append(val)

        df = pd.DataFrame({"alpaca_eval": alpaca_eval_values})
        df.to_csv(f"{csv_save_path}/alpaca_eval_scores_{model_id}.csv")
        print(f"Saved to {csv_save_path}/alpaca_eval_scores_{model_id}.csv")
