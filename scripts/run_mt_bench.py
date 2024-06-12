import argparse
import os
import subprocess
from datetime import datetime

import pandas as pd


def evaluate_gpo(model_id, model_path, config):
    print("Running MT Bench")
    command = [
        "python",
        "gen_model_answer.py",
        "--model-path",
        model_path,
        "--model-id",
        model_id,
        "--max-new-token",
        str(config["MAX_NEW_TOKEN"]),
        "--num-gpus-total",
        str(config["NUM_GPUS"]),
        "--num-choices",
        str(config["NUM_GENERATIONS"]),
    ]

    cwd = "../FastChat/fastchat/llm_judge"
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        return (
            False,
            f"Gen Model failed with return code: {result.returncode}\n{result.stderr}",
        )

    command = [
        "python",
        "gen_judgment.py",
        "--model-list",
        model_id,
        "--parallel",
        "4",
        #'--first-n', '3',
    ]
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        return (
            False,
            f"Gen Judgemnt failed with return code: {result.returncode}\n{result.stderr}",
        )
    input_file = "../FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl"

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]
    df = df[df["model"].isin([model_id])]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    print("\n########## Second turn ##########")
    df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
    print(df_2.sort_values(by="score", ascending=False))

    print("\n########## Average MT-Bench ##########")
    df_3 = df[["model", "score"]].groupby(["model"]).mean()
    print(df_3.sort_values(by="score", ascending=False))
    mt_bench_score = df_3.loc[model_id]["score"]

    return True, mt_bench_score


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-generations",
    type=int,
    default=5,
    help="Number of times the evaluation is to run",
)
parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
parser.add_argument(
    "--model-id",
    type=str,
    help="Your personal model ID, does not have to match the model path name",
)
parser.add_argument(
    "--model-path",
    type=str,
    help="Model path, either to directory where the weights are saved or to a HF repo",
)
parser.add_argument("--mt-bench", action="store_true", help="Whether to evaluate on MT Bench")
parser.add_argument("--no-logging", action="store_true")
parser.add_argument("--csv-save-path", type=str, default="./", help="Path to save the CSV file")
parser.add_argument("--max-new-token", type=int, help="Max number of tokens to generate")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_gpus in [1, 2, 4, 8], "NUM GPUS must be 1, 2, 4 or 8"
    print(f"NUM GPUS: {args.num_gpus}")

    now = str(datetime.now()).replace(" ", "_")
    if not args.no_logging:
        save_dir = f"runs/{now}"
        os.mkdir(save_dir)

    config = {
        "NUM_GENERATIONS": args.num_generations,
        "SAVE_DIR": save_dir,
        "NOW": now,
        "NUM_GPUS": args.num_gpus,
        "MAX_NEW_TOKEN": args.max_new_token,
    }

    mt_bench_values = []
    model_id = args.model_id

    # TODO: DONT HARD CODE PATHS
    csv_save_path = "."

    for i in range(1):  # range(args.num_generations):
        print(f"Iteration: {i}")
        evaluated, val = evaluate_gpo(
            model_id,
            args.model_path,
            config,
        )
        if evaluated:
            mt_bench_values.append(val)

        df = pd.DataFrame({"mt_bench": mt_bench_values})
        df.to_csv(f"{csv_save_path}/mt_bench_scores_{model_id}-{str(args.max_new_token)}.csv")
        print(f"Saved to {csv_save_path}/mt_bench_scores_{model_id}-{str(args.max_new_token)}.csv")
