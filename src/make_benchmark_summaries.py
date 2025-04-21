from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import json
import re
from sklearn.metrics import recall_score, precision_score
import numpy as np
import plotly.express as px


def load_results(file_path):
    results = []

    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    results_df = pd.DataFrame(results).explode(["generated_text", "finish_reason"])

    return results_df


def extract_final_answer(text):
    # find predicted answer in the generated text
    # this is rather restrictive, it only matches a single character on its own line
    # surrounded by the correct <answer> tags

    match = re.search(r"<answer>\s*(\S)\s*</answer>", text, re.DOTALL)

    return match.group(1).strip() if match else "invalid"


def combine_results(results_dict):
    final_dict = {"metric": ["pass@1", "cons@5"]}
    for k, v in results_dict.items():
        final_dict[k] = [get_passat1(v), get_consat5(v)]
        # print(get_passat1(v), get_consat5(v))
        # final_dict[k] = [pass_at_k(v, 5, 1), get_consat5(v)]
        # print(pass_at_k(v, 5, 1), get_consat5(v))
        # raise ValueError

    # Create a tidy DataFrame
    df = pd.DataFrame(final_dict)

    return df


if __name__ == "__main__":

    cli_config = OmegaConf.load("results_path_to_compare.yml")

    metrics_dicts = []

    for results_dir in cli_config.results_dirs:

        # configuration used for the run we're looking at
        run_config = OmegaConf.load(Path(results_dir) / "config.yml")

        model_name = run_config.model_name_readable

        for benchmark_path_str in run_config.benchmarks:

            benchmark_path = Path(benchmark_path_str)

            benchmark_metadata = OmegaConf.load(benchmark_path / "metadata.yml")

            benchmark_name = benchmark_metadata.benchmark_name

            ground_truth_key = benchmark_metadata.ground_truth_key

            results_file = Path(results_dir) / (benchmark_path.stem + "_output.jsonl")

            # this df is already exploded, it's one row per answer (the same question may appear multiple times)
            # columns: problem, prompt, generated_text, finish_reason
            # The `problem` key is a dictionary,
            # with the ground truth answer in problem[ground_truth_key]
            results_df = load_results(results_file)
            results_df["ground_truth"] = results_df["problem"].apply(
                lambda x: str(x[ground_truth_key])
            )
            results_df["prediction"] = results_df["generated_text"].apply(
                extract_final_answer
            )

            # the columns prediction and ground_truth are what we need to compare

            # how often is prediction `invalid`?
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "invalid_frac",
                    "value": (results_df["prediction"] == "invalid").sum()
                    / len(results_df),
                }
            )

            # save the invalid responses so we can look at them later
            results_df[results_df["prediction"] == "invalid"]["generated_text"].to_csv(
                f"benchmark_summaries/invalid_responses/{benchmark_name}_invalid.json"
            )

            # fraction of answers cut because of maximum allowed length exceeded
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "length_exceeded_frac",
                    "value": (results_df["finish_reason"] != "stop").sum()
                    / len(results_df),
                }
            )

            # length of the generated answer
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "answer_length_mean",
                    "value": results_df.assign(
                        generated_text_len=results_df["generated_text"].apply(len)
                    )["generated_text_len"].mean(),
                }
            )

            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "answer_length_std",
                    "value": results_df.assign(
                        generated_text_len=results_df["generated_text"].apply(len)
                    )["generated_text_len"].std(),
                }
            )

            # we can measure consistency of the answer looking at how many different unique answers you get
            # at zero temperature, or if the model is perfectly consistent, this metrics is identically 1
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "unique_answers_mean",
                    "value": results_df.groupby("prompt")["prediction"]
                    .nunique()
                    .mean(),
                }
            )

            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "unique_answers_std",
                    "value": results_df.groupby("prompt")["prediction"].nunique().std(),
                }
            )

            # pass@1:
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "pass@1_mean",
                    "value": results_df.assign(
                        correct=(results_df["prediction"] == results_df["ground_truth"])
                    )
                    .groupby("prompt")["correct"]
                    .mean()
                    .mean(),
                }
            )

            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "pass@1_std",
                    "value": results_df.assign(
                        correct=(results_df["prediction"] == results_df["ground_truth"])
                    )
                    .groupby("prompt")["correct"]
                    .mean()
                    .std(),
                }
            )

            # cons@5 is majority voting
            # we break ties between answers randomly
            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "consensus_accuracy",
                    "value": (
                        results_df.groupby("prompt")["prediction"].agg(
                            lambda x: x.mode().sample(1)
                        )
                        == results_df.groupby("prompt")["ground_truth"].first()
                    ).mean(),
                }
            )

            # we should use micro averaged metrics, with error bars over the temperature.
            # It doesn't make sense to use macro averaged metrics because the "calsses" are the letters identifying the answers, and they are not different in a meaningful way in the benchmarks
            # except in nacc_cog_status, in which the classes are genuinely different and we should be looking at the confusion matrix

            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "precision_micro_mean",
                    "value": np.mean(
                        [
                            precision_score(
                                *(
                                    results_df.groupby("prompt")
                                    .nth(i)[["ground_truth", "prediction"]]
                                    .values.T
                                ),
                                average="micro",
                            )
                            for i in range(run_config.n)
                        ]
                    ),
                }
            )

            metrics_dicts.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "metric": "precision_micro_std",
                    "value": np.std(
                        [
                            precision_score(
                                *(
                                    results_df.groupby("prompt")
                                    .nth(i)[["ground_truth", "prediction"]]
                                    .values.T
                                ),
                                average="micro",
                            )
                            for i in range(run_config.n)
                        ],
                        ddof=1,
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics_dicts)
    metrics_df.to_csv("benchmark_summaries/benchmark_summaries_tall.csv")

    results_wide = metrics_df.pivot_table(
        index=["benchmark", "model"], columns="metric", values="value"
    )
    results_wide.to_csv("benchmark_summaries/benchmark_summaries_wide.csv")